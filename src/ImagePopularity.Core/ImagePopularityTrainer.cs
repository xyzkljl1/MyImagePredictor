using System.Diagnostics;
using System.Globalization;
using TorchSharp;
using static TorchSharp.torch;

namespace ImagePopularity.Core;

public sealed class ImagePopularityTrainer
    : IDisposable
{
    private const double BytesPerGiB = 1024d * 1024d * 1024d;

    private readonly ImagePopularityTrainingOptions _options;
    private readonly Device _device;
    private readonly Random _random;
    private readonly ImageAugmentationOptions? _trainAugmentationOptions;
    private readonly PreprocessedImageCache _trainPreprocessedImageCache;
    private readonly PreprocessedImageCache _validationPreprocessedImageCache;
    private volatile bool _cancellationRequested;
    private bool _disposed;

    public ImagePopularityTrainer(ImagePopularityTrainingOptions options)
    {
        _options = options;

        if (_options.Epochs <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.Epochs), _options.Epochs, "epochs must be > 0.");
        }

        if (_options.BatchSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.BatchSize), _options.BatchSize, "batch-size must be > 0.");
        }

        if (_options.ValidationSplit is <= 0 or >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(options.ValidationSplit), _options.ValidationSplit, "validation-split must be in (0, 1).");
        }

        _device = TorchDeviceSelector.ResolveCuda();
        _random = new Random(options.Seed);
        _trainAugmentationOptions = options.EnableAugmentation
            ? new ImageAugmentationOptions
            {
                Enabled = true,
                HorizontalFlipProbability = (float)options.HorizontalFlipProbability,
                MaxRotationDegrees = (float)options.MaxRotationDegrees,
                BrightnessJitter = (float)options.BrightnessJitter,
                ContrastJitter = (float)options.ContrastJitter,
                SaturationJitter = (float)options.SaturationJitter,
                MinRandomCropScale = (float)options.MinRandomCropScale
            }
            : null;

        var cacheRoot = Path.GetFullPath(_options.PreprocessCacheDirectory);
        _trainPreprocessedImageCache = new PreprocessedImageCache(
            cacheDirectory: Path.Combine(cacheRoot, "train"),
            pipeline: new PreprocessedImageCache.PreprocessPipelineSpec
            {
                ImageSize = _options.TrainImageSize,
                Seed = _options.Seed,
                UseAugmentation = _options.EnableAugmentation,
                AugmentationOptions = _trainAugmentationOptions
            });
        _validationPreprocessedImageCache = new PreprocessedImageCache(
            cacheDirectory: Path.Combine(cacheRoot, "validation"),
            pipeline: new PreprocessedImageCache.PreprocessPipelineSpec
            {
                ImageSize = _options.TrainImageSize,
                Seed = _options.Seed,
                UseAugmentation = false,
                AugmentationOptions = null
            });
    }

    public void Run()
    {
        EnsureNotDisposed();
        var totalStopwatch = Stopwatch.StartNew();
        var bestModelSaved = false;
        var bestModelPromoted = false;

        var (trainSamples, validationSamples) = LoadDatasets();
        if (trainSamples.Count + validationSamples.Count < 100)
        {
            throw new InvalidOperationException("Not enough labeled images. Need at least 100 samples to train a reliable model.");
        }

        ThrowIfCancellationRequested();

        var pretrainedWeightsFile = PretrainedWeightsResolver.Resolve(_options);
        var modelOutputPath = BuildTemporaryAutoOutputModelPath(_options.BuildInProgressOutputModelPath(trainSamples.Count));
        var progressMemoryProvider = CreateGpuMemoryStatusProvider();

        Console.WriteLine($"Preprocess cache directory: {Path.GetFullPath(_options.PreprocessCacheDirectory)}");
        Console.WriteLine("Preparing training preprocess cache (augment + pad resize + normalize)...");
        var trainPreprocessStopwatch = Stopwatch.StartNew();
        var trainCacheBuild = _trainPreprocessedImageCache.Build(
            trainSamples.Select(x => x.ImagePath),
            progressLabel: "Preprocess train",
            dynamicStatusProvider: progressMemoryProvider,
            cancellationRequested: IsCancellationRequested);
        trainPreprocessStopwatch.Stop();
        Console.WriteLine(
            $"Training preprocess cache summary: total={trainCacheBuild.Total}, reused={trainCacheBuild.Reused}, created={trainCacheBuild.Created}, failed={trainCacheBuild.Failed}, elapsed={FormatElapsed(trainPreprocessStopwatch.Elapsed)}");
        Console.WriteLine("Preparing validation preprocess cache (pad resize + normalize)...");
        var validationPreprocessStopwatch = Stopwatch.StartNew();
        var validationCacheBuild = _validationPreprocessedImageCache.Build(
            validationSamples.Select(x => x.ImagePath),
            progressLabel: "Preprocess validation",
            dynamicStatusProvider: progressMemoryProvider,
            cancellationRequested: IsCancellationRequested);
        validationPreprocessStopwatch.Stop();
        Console.WriteLine(
            $"Validation preprocess cache summary: total={validationCacheBuild.Total}, reused={validationCacheBuild.Reused}, created={validationCacheBuild.Created}, failed={validationCacheBuild.Failed}, elapsed={FormatElapsed(validationPreprocessStopwatch.Elapsed)}");

        Console.WriteLine($"Device: {_device.type}");
        Console.WriteLine($"Backbone: {_options.Backbone}");
        Console.WriteLine("Use pretrained backbone: true (forced)");
        Console.WriteLine($"Train image size: {_options.TrainImageSize}");
        Console.WriteLine($"Recommended inference image size: {_options.TrainImageSize} (auto = train-image-size)");
        Console.WriteLine($"Data augmentation enabled: {_options.EnableAugmentation}");
        Console.WriteLine($"Pretrained weights: {pretrainedWeightsFile}");
        var trainPopularCount = trainSamples.Count(x => x.Label > 0.5f);
        var trainUnpopularCount = trainSamples.Count - trainPopularCount;
        var validationPopularCount = validationSamples.Count(x => x.Label > 0.5f);
        var validationUnpopularCount = validationSamples.Count - validationPopularCount;
        Console.WriteLine($"Popular train samples: {trainPopularCount}, Unpopular train samples: {trainUnpopularCount}");
        Console.WriteLine($"Popular validation samples: {validationPopularCount}, Unpopular validation samples: {validationUnpopularCount}");
        Console.WriteLine($"Train samples: {trainSamples.Count}, Validation samples: {validationSamples.Count}");

        if (HasMoreThanTwoTimesImbalance(trainPopularCount, trainUnpopularCount))
        {
            Console.WriteLine(
                $"Warning: training set class imbalance is greater than 2x (popular={trainPopularCount}, unpopular={trainUnpopularCount}). This may bias the model toward the larger class.");
        }

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(modelOutputPath))!);

        using var model = new PopularityModel(
            new PopularityModelConfig { Backbone = _options.Backbone },
            backboneWeightsFile: pretrainedWeightsFile,
            device: CPU);
        model.to(_device);

        if (_options.FreezeBackboneEpochs > 0)
        {
            model.SetBackboneTrainable(false);
            Console.WriteLine($"Backbone frozen for first {_options.FreezeBackboneEpochs} epoch(s).");
        }
        else
        {
            model.SetBackboneTrainable(true);
        }

        var currentLearningRate = _options.LearningRate;
        var optimizer = CreateOptimizer(model, currentLearningRate);

        var bestValidationLoss = double.MaxValue;
        var epochTimings = new List<EpochTiming>(_options.Epochs);

        try
        {
            for (var epoch = 1; epoch <= _options.Epochs; epoch++)
            {
                ThrowIfCancellationRequested();
                var epochStopwatch = Stopwatch.StartNew();
                if (_options.FreezeBackboneEpochs > 0 &&
                    epoch == _options.FreezeBackboneEpochs + 1)
                {
                    model.SetBackboneTrainable(true);
                    optimizer.Dispose();
                    currentLearningRate = _options.FineTuneLearningRate;
                    optimizer = CreateOptimizer(model, currentLearningRate);
                    Console.WriteLine($"Backbone unfrozen at epoch {epoch}. Switched lr to {currentLearningRate:E5}.");
                }

                var trainMetrics = RunEpoch(model, trainSamples, optimizer, isTraining: true, epoch);
                ThrowIfCancellationRequested();
                var valMetrics = RunEpoch(model, validationSamples, optimizer: null, isTraining: false, epoch);
                epochStopwatch.Stop();

                epochTimings.Add(new EpochTiming(
                    epoch,
                    trainMetrics.Duration,
                    valMetrics.Duration,
                    epochStopwatch.Elapsed));

                Console.WriteLine(
                    $"Epoch {epoch}/{_options.Epochs} | " +
                    $"LR={currentLearningRate:E5} | " +
                    $"Train Loss={trainMetrics.Loss:F4}, Train Acc={trainMetrics.Accuracy:P2} | " +
                    $"Val Loss={valMetrics.Loss:F4}, Val Acc={valMetrics.Accuracy:P2} | " +
                    $"Train Time={FormatElapsed(trainMetrics.Duration)} | " +
                    $"Val Time={FormatElapsed(valMetrics.Duration)} | " +
                    $"Epoch Time={FormatElapsed(epochStopwatch.Elapsed)}");

                if (valMetrics.Loss < bestValidationLoss)
                {
                    bestValidationLoss = valMetrics.Loss;
                    model.save(modelOutputPath);
                    bestModelSaved = true;

                    var metadata = new PopularityModelMetadata
                    {
                        TrainingImageSize = _options.TrainImageSize,
                        RecommendedInferenceImageSize = _options.TrainImageSize,
                        Backbone = _options.Backbone,
                        UsedPretrainedBackbone = true,
                        PretrainedWeightsReference = Path.GetFileName(pretrainedWeightsFile),
                        Epochs = epoch,
                        TrainSamples = trainSamples.Count,
                        ValidationSamples = validationSamples.Count,
                        TrainedDevice = _device.type.ToString(),
                        TrainedAtUtc = DateTimeOffset.UtcNow
                    };

                    metadata.Save(modelOutputPath);

                    Console.WriteLine($"Saved best model: {modelOutputPath} (Val Loss={bestValidationLoss:F4})");
                }
            }

            if (bestModelSaved)
            {
                var finalOutputPath = _options.BuildCompletedAutoOutputModelPath(DateTimeOffset.Now, trainSamples.Count);
                PromoteAutoNamedOutput(modelOutputPath, finalOutputPath);
                modelOutputPath = finalOutputPath;
                bestModelPromoted = true;
                Console.WriteLine($"Final auto-named model: {modelOutputPath}");
            }

            totalStopwatch.Stop();
            PrintTimingSummary(
                trainPreprocessStopwatch.Elapsed,
                validationPreprocessStopwatch.Elapsed,
                epochTimings,
                totalStopwatch.Elapsed);
        }
        catch
        {
            totalStopwatch.Stop();

            if (bestModelSaved && !bestModelPromoted)
            {
                bestModelPromoted = !string.IsNullOrWhiteSpace(
                    TryPromoteInterruptedBestModel(modelOutputPath, trainSamples.Count));
            }

            throw;
        }
        finally
        {
            optimizer.Dispose();
        }
    }

    public void RequestCancellation()
    {
        _cancellationRequested = true;
    }

    private EpochMetrics RunEpoch(
        PopularityModel model,
        IReadOnlyList<LabeledImageSample> samples,
        optim.Optimizer? optimizer,
        bool isTraining,
        int epoch)
    {
        var phaseStopwatch = Stopwatch.StartNew();

        if (isTraining)
        {
            model.train();

            // Keep the frozen backbone in eval mode even after model.train()
            // recursively switches child modules back to training mode.
            if (_options.FreezeBackboneEpochs > 0 && epoch <= _options.FreezeBackboneEpochs)
            {
                model.SetBackboneTrainable(false);
            }
        }
        else
        {
            model.eval();
        }

        double totalLoss = 0;
        long totalCorrect = 0;
        long totalCount = 0;

        using var noGrad = isTraining ? null : torch.no_grad();
        var plannedBatches = Math.Max(1, (int)Math.Ceiling(samples.Count / (double)_options.BatchSize));
        var phaseLabel = isTraining ? $"Train E{epoch}/{_options.Epochs}" : $"Valid E{epoch}/{_options.Epochs}";
        using var progress = new ConsoleProgressBar(
            phaseLabel,
            plannedBatches,
            clearOnComplete: true,
            dynamicStatusProvider: CreateGpuMemoryStatusProvider());

        var cache = isTraining ? _trainPreprocessedImageCache : _validationPreprocessedImageCache;
        var processedBatches = 0;
        foreach (var batch in CreateBatches(samples, _options.BatchSize, shuffle: isTraining, cache: cache))
        {
            ThrowIfCancellationRequested();

            try
            {
                using var inputs = batch.Inputs.to(_device);
                using var labels = batch.Labels.to(_device);

                if (isTraining)
                {
                    optimizer!.zero_grad();
                }

                using var logits = model.forward(inputs);
                using var loss = nn.functional.binary_cross_entropy_with_logits(logits, labels);

                if (isTraining)
                {
                    loss.backward();
                    optimizer!.step();
                }

                var batchSize = labels.shape[0];
                totalLoss += loss.item<float>() * (double)batchSize;

                using var probabilities = torch.sigmoid(logits);
                using var predictions = torch.ge(probabilities, 0.5);
                using var expected = torch.ge(labels, 0.5);
                using var correct = torch.eq(predictions, expected);

                totalCorrect += correct.sum().item<long>();
                totalCount += batchSize;
                processedBatches++;

                var avgLoss = totalCount == 0 ? 0 : totalLoss / totalCount;
                var avgAccuracy = totalCount == 0 ? 0 : totalCorrect / (double)totalCount;
                progress.Report(processedBatches, $"loss={avgLoss:F4} acc={avgAccuracy:P2}");
            }
            finally
            {
                batch.Inputs.Dispose();
                batch.Labels.Dispose();
            }
        }

        var metrics = new EpochMetrics
        {
            Loss = totalCount == 0 ? double.MaxValue : totalLoss / totalCount,
            Accuracy = totalCount == 0 ? 0 : totalCorrect / (double)totalCount,
            Duration = phaseStopwatch.Elapsed
        };
        progress.Complete($"loss={metrics.Loss:F4} acc={metrics.Accuracy:P2}");

        return metrics;
    }

    private IEnumerable<TrainingBatch> CreateBatches(
        IReadOnlyList<LabeledImageSample> samples,
        int batchSize,
        bool shuffle,
        PreprocessedImageCache cache)
    {
        var ordered = samples.ToList();
        if (shuffle)
        {
            ShuffleInPlace(ordered);
        }

        var imageTensorSize = 3 * _options.TrainImageSize * _options.TrainImageSize;

        for (var start = 0; start < ordered.Count; start += batchSize)
        {
            ThrowIfCancellationRequested();

            var count = Math.Min(batchSize, ordered.Count - start);
            var features = new float[count * imageTensorSize];
            var labels = new float[count];

            var valid = 0;

            for (var i = 0; i < count; i++)
            {
                var sample = ordered[start + i];

                try
                {
                    var chw = cache.LoadChw(sample.ImagePath);
                    Array.Copy(chw, 0, features, valid * imageTensorSize, imageTensorSize);
                    labels[valid] = sample.Label;
                    valid++;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Skip broken image: {sample.ImagePath} ({ex.Message})");
                }
            }

            if (valid == 0)
            {
                continue;
            }

            if (valid < count)
            {
                Array.Resize(ref features, valid * imageTensorSize);
                Array.Resize(ref labels, valid);
            }

            var inputTensor = torch.tensor(features, dtype: ScalarType.Float32)
                .reshape(valid, 3, _options.TrainImageSize, _options.TrainImageSize);
            var labelTensor = torch.tensor(labels, dtype: ScalarType.Float32)
                .reshape(valid, 1);

            yield return new TrainingBatch(inputTensor, labelTensor);
        }
    }

    private (IReadOnlyList<LabeledImageSample> Train, IReadOnlyList<LabeledImageSample> Validation) LoadDatasets()
    {
        if (_options.ValidationDirectories.Count == 0)
        {
            var allSamples = LoadSamplesForRandomSplit();
            return SplitStratified(allSamples, _options.ValidationSplit);
        }

        return LoadSamplesFromExplicitValidationDirectories(_options.ValidationDirectories);
    }

    private IReadOnlyList<LabeledImageSample> LoadSamplesForRandomSplit()
    {
        var popular = EnumerateImages(_options.PopularDirectory)
            .Select(path => new LabeledImageSample(path, 1f))
            .ToList();

        var unpopular = EnumerateImages(_options.UnpopularDirectory)
            .Select(path => new LabeledImageSample(path, 0f))
            .ToList();

        if (_options.MaxSamplesPerClass > 0)
        {
            popular = popular
                .OrderBy(_ => _random.Next())
                .Take(_options.MaxSamplesPerClass)
                .ToList();

            unpopular = unpopular
                .OrderBy(_ => _random.Next())
                .Take(_options.MaxSamplesPerClass)
                .ToList();
        }

        if (popular.Count == 0 || unpopular.Count == 0)
        {
            throw new InvalidOperationException("Both popular and unpopular directories must contain images.");
        }

        Console.WriteLine($"Popular images: {popular.Count}");
        Console.WriteLine($"Unpopular images: {unpopular.Count}");
        Console.WriteLine($"Validation selection: random stratified split ({_options.ValidationSplit:P0})");

        return [.. popular, .. unpopular];
    }

    private (IReadOnlyList<LabeledImageSample> Train, IReadOnlyList<LabeledImageSample> Validation) LoadSamplesFromExplicitValidationDirectories(IReadOnlyList<string> validationDirectories)
    {
        var popularValidationDirectories = ResolveExistingValidationDirectories(_options.PopularDirectory, validationDirectories);
        var unpopularValidationDirectories = ResolveExistingValidationDirectories(_options.UnpopularDirectory, validationDirectories);

        var trainPopular = EnumerateImages(_options.PopularDirectory, excludedDirectories: popularValidationDirectories)
            .Select(path => new LabeledImageSample(path, 1f))
            .ToList();
        var trainUnpopular = EnumerateImages(_options.UnpopularDirectory, excludedDirectories: unpopularValidationDirectories)
            .Select(path => new LabeledImageSample(path, 0f))
            .ToList();

        if (_options.MaxSamplesPerClass > 0)
        {
            trainPopular = trainPopular
                .OrderBy(_ => _random.Next())
                .Take(_options.MaxSamplesPerClass)
                .ToList();

            trainUnpopular = trainUnpopular
                .OrderBy(_ => _random.Next())
                .Take(_options.MaxSamplesPerClass)
                .ToList();
        }

        var validationPopular = EnumerateImagesFromDirectories(popularValidationDirectories)
            .Select(path => new LabeledImageSample(path, 1f))
            .ToList();
        var validationUnpopular = EnumerateImagesFromDirectories(unpopularValidationDirectories)
            .Select(path => new LabeledImageSample(path, 0f))
            .ToList();

        if (trainPopular.Count == 0 || trainUnpopular.Count == 0)
        {
            throw new InvalidOperationException("Explicit validation directories left no training images in one or both classes.");
        }

        var validationCount = validationPopular.Count + validationUnpopular.Count;
        var totalCount = trainPopular.Count + trainUnpopular.Count + validationCount;
        var minimumValidationCount = (int)Math.Ceiling(totalCount * _options.ValidationSplit);
        if (validationCount < minimumValidationCount)
        {
            throw new InvalidOperationException(
                $"Explicit validation set is too small: found {validationCount} image(s), but at least {minimumValidationCount} are required by validation-split={_options.ValidationSplit.ToString("0.####", System.Globalization.CultureInfo.InvariantCulture)}.");
        }

        Console.WriteLine($"Popular train images: {trainPopular.Count}");
        Console.WriteLine($"Unpopular train images: {trainUnpopular.Count}");
        Console.WriteLine($"Popular validation images: {validationPopular.Count}");
        Console.WriteLine($"Unpopular validation images: {validationUnpopular.Count}");
        Console.WriteLine($"Validation selection: explicit subdirectories [{string.Join(", ", validationDirectories)}]");
        Console.WriteLine($"Existing popular validation directories: {FormatDirectoryList(popularValidationDirectories)}");
        Console.WriteLine($"Existing unpopular validation directories: {FormatDirectoryList(unpopularValidationDirectories)}");

        var train = trainPopular
            .Concat(trainUnpopular)
            .OrderBy(_ => _random.Next())
            .ToList();
        var validation = validationPopular
            .Concat(validationUnpopular)
            .OrderBy(_ => _random.Next())
            .ToList();

        return (train, validation);
    }

    private static IEnumerable<string> EnumerateImagesFromDirectories(IReadOnlyList<string> directories)
    {
        return directories
            .SelectMany(directory => EnumerateImagesStatic(directory))
            .Distinct(StringComparer.OrdinalIgnoreCase);
    }

    private IEnumerable<string> EnumerateImages(string directory, IReadOnlyList<string>? excludedDirectories = null)
    {
        if (!Directory.Exists(directory))
        {
            throw new DirectoryNotFoundException($"Directory not found: {directory}");
        }

        var normalizedExcluded = excludedDirectories?
            .Where(path => !string.IsNullOrWhiteSpace(path))
            .Select(NormalizeDirectoryPath)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();

        return Directory
            .EnumerateFiles(directory, "*", SearchOption.AllDirectories)
            .Where(SupportedImageFiles.IsSupported)
            .Where(path => normalizedExcluded is null || !IsUnderAnyDirectory(path, normalizedExcluded));
    }

    private static IEnumerable<string> EnumerateImagesStatic(string directory)
    {
        if (!Directory.Exists(directory))
        {
            return Array.Empty<string>();
        }

        return Directory
            .EnumerateFiles(directory, "*", SearchOption.AllDirectories)
            .Where(SupportedImageFiles.IsSupported);
    }

    private (IReadOnlyList<LabeledImageSample> Train, IReadOnlyList<LabeledImageSample> Validation) SplitStratified(
        IReadOnlyList<LabeledImageSample> all,
        double validationSplit)
    {
        var popular = all.Where(x => x.Label > 0.5f).OrderBy(_ => _random.Next()).ToList();
        var unpopular = all.Where(x => x.Label < 0.5f).OrderBy(_ => _random.Next()).ToList();

        var popularValidationCount = Math.Max(1, (int)Math.Round(popular.Count * validationSplit));
        var unpopularValidationCount = Math.Max(1, (int)Math.Round(unpopular.Count * validationSplit));

        var validation = popular.Take(popularValidationCount)
            .Concat(unpopular.Take(unpopularValidationCount))
            .OrderBy(_ => _random.Next())
            .ToList();

        var train = popular.Skip(popularValidationCount)
            .Concat(unpopular.Skip(unpopularValidationCount))
            .OrderBy(_ => _random.Next())
            .ToList();

        return (train, validation);
    }

    private void ShuffleInPlace<T>(IList<T> values)
    {
        for (var i = values.Count - 1; i > 0; i--)
        {
            var j = _random.Next(i + 1);
            (values[i], values[j]) = (values[j], values[i]);
        }
    }

    private optim.Optimizer CreateOptimizer(PopularityModel model, double learningRate)
    {
        var trainableParameters = model.parameters()
            .Where(p => p.requires_grad)
            .ToArray();

        if (trainableParameters.Length == 0)
        {
            throw new InvalidOperationException("No trainable parameters found for optimizer.");
        }

        return optim.AdamW(trainableParameters, lr: learningRate, weight_decay: _options.WeightDecay);
    }

    private static void PrintTimingSummary(
        TimeSpan trainPreprocessDuration,
        TimeSpan validationPreprocessDuration,
        IReadOnlyList<EpochTiming> epochTimings,
        TimeSpan totalDuration)
    {
        Console.WriteLine("Timing summary:");
        Console.WriteLine($"  Preprocess train: {FormatElapsed(trainPreprocessDuration)}");
        Console.WriteLine($"  Preprocess validation: {FormatElapsed(validationPreprocessDuration)}");

        foreach (var epochTiming in epochTimings)
        {
            Console.WriteLine(
                $"  Epoch {epochTiming.Epoch}: train={FormatElapsed(epochTiming.TrainDuration)}, " +
                $"validation={FormatElapsed(epochTiming.ValidationDuration)}, total={FormatElapsed(epochTiming.TotalDuration)}");
        }

        var preprocessTotal = trainPreprocessDuration + validationPreprocessDuration;
        var epochTotal = TimeSpan.FromTicks(epochTimings.Sum(x => x.TotalDuration.Ticks));
        var otherDuration = totalDuration - preprocessTotal - epochTotal;
        if (otherDuration < TimeSpan.Zero)
        {
            otherDuration = TimeSpan.Zero;
        }

        Console.WriteLine($"  Other overhead: {FormatElapsed(otherDuration)}");
        Console.WriteLine($"  Total: {FormatElapsed(totalDuration)}");
    }

    private static string FormatElapsed(TimeSpan elapsed)
    {
        if (elapsed.TotalHours >= 1)
        {
            return elapsed.ToString(@"hh\:mm\:ss");
        }

        return elapsed.ToString(@"mm\:ss");
    }

    private static string BuildTemporaryAutoOutputModelPath(string baseOutputPath)
    {
        var directory = Path.GetDirectoryName(baseOutputPath);
        var extension = Path.GetExtension(baseOutputPath);
        var fileNameWithoutExtension = Path.GetFileNameWithoutExtension(baseOutputPath);
        var temporaryFileName = string.IsNullOrEmpty(extension)
            ? $"{fileNameWithoutExtension}.autosave.tmp"
            : $"{fileNameWithoutExtension}.autosave{extension}";

        return string.IsNullOrWhiteSpace(directory)
            ? temporaryFileName
            : Path.Combine(directory, temporaryFileName);
    }

    private static void PromoteAutoNamedOutput(string sourceModelPath, string destinationModelPath)
    {
        var destinationDirectory = Path.GetDirectoryName(Path.GetFullPath(destinationModelPath));
        if (!string.IsNullOrWhiteSpace(destinationDirectory))
        {
            Directory.CreateDirectory(destinationDirectory);
        }

        File.Move(sourceModelPath, destinationModelPath, overwrite: true);

        var sourceMetadataPath = PopularityModelMetadata.GetMetadataPath(sourceModelPath);
        if (!File.Exists(sourceMetadataPath))
        {
            return;
        }

        var destinationMetadataPath = PopularityModelMetadata.GetMetadataPath(destinationModelPath);
        File.Move(sourceMetadataPath, destinationMetadataPath, overwrite: true);
    }

    private string? TryPromoteInterruptedBestModel(string autosaveModelPath, int trainSampleCount)
    {
        try
        {
            if (!File.Exists(autosaveModelPath))
            {
                return null;
            }

            var finalOutputPath = _options.BuildCompletedAutoOutputModelPath(DateTimeOffset.Now, trainSampleCount);
            PromoteAutoNamedOutput(autosaveModelPath, finalOutputPath);
            Console.WriteLine($"Promoted interrupted best model: {finalOutputPath}");
            return finalOutputPath;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Warning: failed to promote interrupted best model from autosave ({ex.Message})");
            return null;
        }
    }

    private static IReadOnlyList<string> ResolveExistingValidationDirectories(string classRoot, IReadOnlyList<string> validationDirectories)
    {
        var existingDirectories = new List<string>();
        foreach (var validationDirectory in validationDirectories)
        {
            var resolved = TryResolveValidationDirectory(classRoot, validationDirectory);
            if (!string.IsNullOrWhiteSpace(resolved))
            {
                existingDirectories.Add(resolved);
            }
        }

        return existingDirectories
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }

    private static string? TryResolveValidationDirectory(string classRoot, string validationDirectory)
    {
        var normalizedRoot = NormalizeDirectoryPath(classRoot);
        var resolved = NormalizeDirectoryPath(Path.GetFullPath(Path.Combine(normalizedRoot, validationDirectory)));

        if (!IsUnderDirectory(resolved, normalizedRoot))
        {
            throw new InvalidOperationException($"validation-dir must stay inside class root. Root={classRoot}, validation-dir={validationDirectory}");
        }

        if (!Directory.Exists(resolved))
        {
            return null;
        }

        return resolved;
    }

    private static string FormatDirectoryList(IReadOnlyList<string> directories)
    {
        return directories.Count == 0
            ? "(none)"
            : string.Join(", ", directories);
    }

    private static bool IsUnderAnyDirectory(string filePath, IReadOnlyList<string> normalizedDirectories)
    {
        foreach (var directory in normalizedDirectories)
        {
            if (IsUnderDirectory(filePath, directory))
            {
                return true;
            }
        }

        return false;
    }

    private static bool IsUnderDirectory(string path, string normalizedDirectory)
    {
        var normalizedPath = Path.GetFullPath(path).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        if (OperatingSystem.IsWindows())
        {
            normalizedPath = normalizedPath.ToUpperInvariant();
        }

        return normalizedPath.Length > normalizedDirectory.Length &&
               normalizedPath.StartsWith(normalizedDirectory, StringComparison.OrdinalIgnoreCase) &&
               (normalizedPath[normalizedDirectory.Length] == Path.DirectorySeparatorChar ||
                normalizedPath[normalizedDirectory.Length] == Path.AltDirectorySeparatorChar);
    }

    private static string NormalizeDirectoryPath(string directory)
    {
        var normalized = Path.GetFullPath(directory).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        if (OperatingSystem.IsWindows())
        {
            normalized = normalized.ToUpperInvariant();
        }

        return normalized;
    }

    private static bool HasMoreThanTwoTimesImbalance(int firstCount, int secondCount)
    {
        if (firstCount <= 0 || secondCount <= 0)
        {
            return false;
        }

        var larger = Math.Max(firstCount, secondCount);
        var smaller = Math.Min(firstCount, secondCount);
        return larger > smaller * 2;
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _trainPreprocessedImageCache.Dispose();
        _validationPreprocessedImageCache.Dispose();
        _disposed = true;
    }

    private void EnsureNotDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(ImagePopularityTrainer));
        }
    }

    private bool IsCancellationRequested()
    {
        return _cancellationRequested;
    }

    private void ThrowIfCancellationRequested()
    {
        if (_cancellationRequested)
        {
            throw new OperationCanceledException("Training was canceled.");
        }
    }

    private Func<string?> CreateGpuMemoryStatusProvider()
    {
        var deviceIndex = TryGetCudaDeviceIndex(_device);
        var cache = new GpuMemoryStatusCache(deviceIndex);
        return cache.GetStatus;
    }

    private static int? TryGetCudaDeviceIndex(Device device)
    {
        try
        {
            var property = typeof(Device).GetProperty("index");
            if (property?.GetValue(device) is int intIndex)
            {
                return intIndex;
            }

            if (property?.GetValue(device) is long longIndex)
            {
                return checked((int)longIndex);
            }
        }
        catch
        {
        }

        return null;
    }

    private readonly record struct LabeledImageSample(string ImagePath, float Label);

    private readonly record struct TrainingBatch(Tensor Inputs, Tensor Labels);

    private sealed class EpochMetrics
    {
        public double Loss { get; init; }

        public double Accuracy { get; init; }

        public TimeSpan Duration { get; init; }
    }

    private readonly record struct EpochTiming(int Epoch, TimeSpan TrainDuration, TimeSpan ValidationDuration, TimeSpan TotalDuration);

    private sealed class GpuMemoryStatusCache
    {
        private static readonly TimeSpan RefreshInterval = TimeSpan.FromSeconds(1);

        private readonly int? _deviceIndex;
        private readonly Stopwatch _stopwatch = Stopwatch.StartNew();
        private string? _lastStatus;

        public GpuMemoryStatusCache(int? deviceIndex)
        {
            _deviceIndex = deviceIndex;
        }

        public string? GetStatus()
        {
            if (_stopwatch.Elapsed < RefreshInterval && _lastStatus is not null)
            {
                return _lastStatus;
            }

            _stopwatch.Restart();
            _lastStatus = QueryGpuMemoryStatus(_deviceIndex) ?? _lastStatus;
            return _lastStatus;
        }

        private static string? QueryGpuMemoryStatus(int? deviceIndex)
        {
            try
            {
                using var process = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = "nvidia-smi",
                        Arguments = "--query-gpu=memory.used --format=csv,noheader,nounits",
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        UseShellExecute = false,
                        CreateNoWindow = true
                    }
                };

                if (!process.Start())
                {
                    return null;
                }

                if (!process.WaitForExit(1500))
                {
                    try
                    {
                        process.Kill(entireProcessTree: true);
                    }
                    catch
                    {
                    }

                    return null;
                }

                if (process.ExitCode != 0)
                {
                    return null;
                }

                var lines = process.StandardOutput
                    .ReadToEnd()
                    .Split(['\r', '\n'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

                if (lines.Length == 0)
                {
                    return null;
                }

                var lineIndex = deviceIndex is >= 0 and < int.MaxValue && deviceIndex.Value < lines.Length
                    ? deviceIndex.Value
                    : 0;

                if (!double.TryParse(lines[lineIndex], NumberStyles.Float, CultureInfo.InvariantCulture, out var usedMiB))
                {
                    return null;
                }

                return $"vram={usedMiB / 1024d:F2}GB";
            }
            catch
            {
                return null;
            }
        }
    }
}
