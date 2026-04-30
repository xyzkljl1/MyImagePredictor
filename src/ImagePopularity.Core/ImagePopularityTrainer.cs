using System.Diagnostics;
using System.Globalization;
using System.Runtime.ExceptionServices;
using TorchSharp;
using static TorchSharp.torch;

namespace ImagePopularity.Core;

public sealed class ImagePopularityTrainer
    : IDisposable
{
    private const double BytesPerGiB = 1024d * 1024d * 1024d;
    private const double GroupWeightExponent = 0.5d;
    private const int MaxSameGroupPerTrainingBatch = 2;
    private const double ThresholdScanMinimum = 0.20d;
    private const double ThresholdScanMaximum = 0.70d;
    private const double ThresholdScanStep = 0.01d;
    private const double PopularLossTarget = 0.5d;

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
        ExceptionDispatchInfo? capturedException = null;
        string? pendingFinalOutputPath = null;
        var bestSavedSelection = LossPriorityState.Unset;
        var bestEarlyStoppingSelection = LossPriorityState.Unset;
        double bestSavedDecisionThreshold = _options.DecisionThreshold;

        var (trainSamples, validationSamples) = LoadDatasets();
        if (trainSamples.Count + validationSamples.Count < 100)
        {
            throw new InvalidOperationException("Not enough labeled images. Need at least 100 samples to train a reliable model.");
        }

        ThrowIfCancellationRequested();

        var pretrainedWeightsFile = PretrainedWeightsResolver.Resolve(_options);
        var modelOutputPath = BuildTemporaryAutoOutputModelPath(_options.BuildInProgressOutputModelPath(trainSamples.Count));
        using var progressMemorySampler = CreateGpuMemoryStatusSampler();
        Func<string?>? progressMemoryProvider = progressMemorySampler is null
            ? null
            : () => progressMemorySampler.GetStatus();

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
        Console.WriteLine("Use pretrained backbone: requested (ConvNeXt official .pth weights are used when compatible; otherwise training falls back to random initialization with a warning).");
        Console.WriteLine($"Train image size: {_options.TrainImageSize}");
        Console.WriteLine($"Recommended inference image size: {_options.TrainImageSize} (auto = train-image-size)");
        Console.WriteLine($"Decision threshold: {_options.DecisionThreshold.ToString("0.##", CultureInfo.InvariantCulture)}");
        Console.WriteLine($"Data augmentation enabled: {_options.EnableAugmentation}");
        Console.WriteLine($"Pretrained weights: {pretrainedWeightsFile}");
        Console.WriteLine("Training batch strategy: balanced P/U batches (minority oversampled when needed).");
        Console.WriteLine("Fine-tune strategy: progressive backbone unfreezing with layer-wise learning-rate scaling.");
        Console.WriteLine(
            $"Best-model objective: prefer Popular Loss < {PopularLossTarget.ToString("0.##", CultureInfo.InvariantCulture)}, then minimize Unpopular Loss, then Val Loss.");
        Console.WriteLine(
            _options.EnableGroupAwareTraining
                ? $"Group-aware training: enabled (group split, group down-weighting 1/n^{GroupWeightExponent.ToString("0.##", CultureInfo.InvariantCulture)}, max {MaxSameGroupPerTrainingBatch} sample(s) per group per batch)."
                : "Group-aware training: disabled.");
        var trainPopularCount = trainSamples.Count(x => x.Label > 0.5f);
        var trainUnpopularCount = trainSamples.Count - trainPopularCount;
        var validationPopularCount = validationSamples.Count(x => x.Label > 0.5f);
        var validationUnpopularCount = validationSamples.Count - validationPopularCount;
        var trainPopularGroupCount = CountDistinctGroups(trainSamples.Where(x => x.Label > 0.5f));
        var trainUnpopularGroupCount = CountDistinctGroups(trainSamples.Where(x => x.Label < 0.5f));
        var validationPopularGroupCount = CountDistinctGroups(validationSamples.Where(x => x.Label > 0.5f));
        var validationUnpopularGroupCount = CountDistinctGroups(validationSamples.Where(x => x.Label < 0.5f));
        Console.WriteLine($"Popular train samples: {trainPopularCount} (groups: {trainPopularGroupCount}), Unpopular train samples: {trainUnpopularCount} (groups: {trainUnpopularGroupCount})");
        Console.WriteLine($"Popular validation samples: {validationPopularCount} (groups: {validationPopularGroupCount}), Unpopular validation samples: {validationUnpopularCount} (groups: {validationUnpopularGroupCount})");
        Console.WriteLine($"Train samples: {trainSamples.Count} (groups: {CountDistinctGroups(trainSamples)}), Validation samples: {validationSamples.Count} (groups: {CountDistinctGroups(validationSamples)})");

        if (HasMoreThanTwoTimesImbalance(trainPopularCount, trainUnpopularCount))
        {
            Console.WriteLine(
                $"Warning: training set class imbalance is greater than 2x (popular={trainPopularCount}, unpopular={trainUnpopularCount}). This may bias the model toward the larger class.");
        }

        var earlyStoppingStartEpoch = Math.Max(2, _options.FreezeBackboneEpochs + 2);
        Console.WriteLine(
            $"Early stopping: enabled (before Popular Loss < {PopularLossTarget.ToString("0.##", CultureInfo.InvariantCulture)} monitor Popular Loss, then monitor Unpopular Loss with patience {_options.EarlyStoppingPatience}, min delta {_options.EarlyStoppingMinDelta.ToString("0.####", CultureInfo.InvariantCulture)}, start epoch {earlyStoppingStartEpoch})");

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(modelOutputPath))!);

        var epochsWithoutMeaningfulImprovement = 0;
        var epochTimings = new List<EpochTiming>(_options.Epochs);

        {
            using var model = new PopularityModel(
                new PopularityModelConfig { Backbone = _options.Backbone },
                backboneWeightsFile: pretrainedWeightsFile,
                device: CPU);
            model.to(_device);

            if (_options.FreezeBackboneEpochs > 0)
            {
                model.ConfigureBackboneTrainability(0);
                Console.WriteLine($"Backbone frozen for first {_options.FreezeBackboneEpochs} epoch(s).");
            }
            else
            {
                model.ConfigureBackboneTrainability(model.BackboneStageCount > 0 ? model.BackboneStageCount : 1);
                Console.WriteLine("Backbone starts fully trainable because freeze-backbone-epochs is 0.");
            }

            var currentLearningRate = _options.LearningRate;
            var currentTrainableBackboneStages = _options.FreezeBackboneEpochs > 0
                ? 0
                : (model.BackboneStageCount > 0 ? model.BackboneStageCount : 1);
            var layerwiseParameterGroups = model.GetLayerwiseParameterGroups(currentTrainableBackboneStages);
            optim.Optimizer? optimizer = null;

            try
            {
                optimizer = CreateOptimizer(layerwiseParameterGroups, currentLearningRate);

                for (var epoch = 1; epoch <= _options.Epochs; epoch++)
                {
                    ThrowIfCancellationRequested();
                    var epochStopwatch = Stopwatch.StartNew();
                    var desiredTrainableBackboneStages = GetTrainableBackboneStageCountForEpoch(model, epoch);
                    var learningRateChanged = false;

                    if (_options.FreezeBackboneEpochs > 0 &&
                        epoch == _options.FreezeBackboneEpochs + 1)
                    {
                        currentLearningRate = _options.FineTuneLearningRate;
                        learningRateChanged = true;
                    }

                    if (desiredTrainableBackboneStages != currentTrainableBackboneStages || learningRateChanged)
                    {
                        currentTrainableBackboneStages = desiredTrainableBackboneStages;
                        model.ConfigureBackboneTrainability(currentTrainableBackboneStages);
                        layerwiseParameterGroups = model.GetLayerwiseParameterGroups(currentTrainableBackboneStages);
                        optimizer.Dispose();
                        optimizer = CreateOptimizer(layerwiseParameterGroups, currentLearningRate);

                        var stageSummary = FormatStageSummary(model.GetTrainableBackboneStageNames(currentTrainableBackboneStages));
                        var scaleSummary = FormatLayerwiseScaleSummary(layerwiseParameterGroups);

                        if (learningRateChanged)
                        {
                            Console.WriteLine(
                                $"Backbone unfrozen at epoch {epoch}. Trainable stages: {stageSummary}. Base lr={FormatLearningRate(currentLearningRate)}. Scales: {scaleSummary}.");
                        }
                        else
                        {
                            Console.WriteLine(
                                $"Expanded backbone training at epoch {epoch}. Trainable stages: {stageSummary}. Base lr={FormatLearningRate(currentLearningRate)}. Scales: {scaleSummary}.");
                        }
                    }
                    else if (epoch == 1)
                    {
                        var stageSummary = FormatStageSummary(model.GetTrainableBackboneStageNames(currentTrainableBackboneStages));
                        var scaleSummary = FormatLayerwiseScaleSummary(layerwiseParameterGroups);
                        Console.WriteLine($"Initial trainable stages: {stageSummary}. Base lr={FormatLearningRate(currentLearningRate)}. Scales: {scaleSummary}.");
                    }

                    var trainMetrics = RunEpoch(
                        model,
                        trainSamples,
                        optimizer,
                        layerwiseParameterGroups,
                        isTraining: true,
                        epoch,
                        trainableBackboneStageCount: currentTrainableBackboneStages,
                        progressMemoryProvider: progressMemoryProvider);
                    ThrowIfCancellationRequested();
                    var valMetrics = RunEpoch(
                        model,
                        validationSamples,
                        optimizer: null,
                        layerwiseParameterGroups: Array.Empty<PopularityModel.LayerwiseParameterGroup>(),
                        isTraining: false,
                        epoch,
                        trainableBackboneStageCount: currentTrainableBackboneStages,
                        progressMemoryProvider: progressMemoryProvider);
                    epochStopwatch.Stop();

                    epochTimings.Add(new EpochTiming(
                        epoch,
                        trainMetrics.Duration,
                        valMetrics.Duration,
                        epochStopwatch.Elapsed));

                    var validationBreakdownText = valMetrics.HasClassBreakdown
                        ? $" | L(P/U) = {valMetrics.PopularLoss:F4}/{valMetrics.UnpopularLoss:F4}, A(P/U) = {valMetrics.PopularAccuracy:P2}/{valMetrics.UnpopularAccuracy:P2}"
                        : string.Empty;
                    var thresholdSummaryText = !double.IsNaN(valMetrics.DecisionThresholdUsed)
                        ? $" | Thr={valMetrics.DecisionThresholdUsed.ToString("0.##", CultureInfo.InvariantCulture)} | BLoss={valMetrics.BalancedLoss:F4}"
                        : string.Empty;

                    Console.WriteLine(
                        $"Epoch {epoch}/{_options.Epochs} | " +
                        $"LR={FormatLearningRate(currentLearningRate)} | " +
                        $"Train Loss={trainMetrics.Loss:F4}, Train Acc={trainMetrics.Accuracy:P2} | " +
                        $"Val Loss={valMetrics.Loss:F4}, Val Acc={valMetrics.Accuracy:P2}" +
                        $"{validationBreakdownText}" +
                        $"{thresholdSummaryText} | " +
                        $"Time(T/V/E)={FormatElapsed(trainMetrics.Duration)}/{FormatElapsed(valMetrics.Duration)}/{FormatElapsed(epochStopwatch.Elapsed)}");

                    if (IsBetterBestModelCandidate(valMetrics, bestSavedSelection))
                    {
                        bestSavedSelection = LossPriorityState.FromEpochMetrics(valMetrics);
                        bestSavedDecisionThreshold = valMetrics.DecisionThresholdUsed;
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
                            DecisionThreshold = bestSavedDecisionThreshold,
                            TrainedDevice = _device.type.ToString(),
                            TrainedAtUtc = DateTimeOffset.UtcNow
                        };

                        metadata.Save(modelOutputPath);

                        Console.WriteLine(
                            $"Saved best model: {modelOutputPath} (Pop Loss={bestSavedSelection.PopularLoss:F4}, Unpop Loss={bestSavedSelection.UnpopularLoss:F4}, Val Loss={bestSavedSelection.TotalLoss:F4}, Thr={bestSavedDecisionThreshold.ToString("0.##", CultureInfo.InvariantCulture)})");
                    }

                    if (HasMeaningfulEarlyStoppingImprovement(valMetrics, bestEarlyStoppingSelection, _options.EarlyStoppingMinDelta))
                    {
                        bestEarlyStoppingSelection = LossPriorityState.FromEpochMetrics(valMetrics);
                        epochsWithoutMeaningfulImprovement = 0;
                    }
                    else if (_options.EarlyStoppingPatience > 0 && epoch >= earlyStoppingStartEpoch)
                    {
                        epochsWithoutMeaningfulImprovement++;
                        if (epochsWithoutMeaningfulImprovement >= _options.EarlyStoppingPatience)
                        {
                            Console.WriteLine(
                                bestEarlyStoppingSelection.MeetsPopularTarget
                                    ? $"Early stopping triggered at epoch {epoch}: no Unpopular Loss improvement >= {_options.EarlyStoppingMinDelta.ToString("0.####", CultureInfo.InvariantCulture)} while keeping Popular Loss < {PopularLossTarget.ToString("0.##", CultureInfo.InvariantCulture)} for {_options.EarlyStoppingPatience} epoch(s)."
                                    : $"Early stopping triggered at epoch {epoch}: no Popular Loss improvement >= {_options.EarlyStoppingMinDelta.ToString("0.####", CultureInfo.InvariantCulture)} and Popular Loss has not yet dropped below {PopularLossTarget.ToString("0.##", CultureInfo.InvariantCulture)} for {_options.EarlyStoppingPatience} epoch(s).");
                            break;
                        }
                    }
                }

                if (bestModelSaved)
                {
                    pendingFinalOutputPath = _options.BuildCompletedAutoOutputModelPath(DateTimeOffset.Now, trainSamples.Count, bestSavedDecisionThreshold);
                }

                totalStopwatch.Stop();
            }
            catch (Exception ex)
            {
                totalStopwatch.Stop();

                if (bestModelSaved && pendingFinalOutputPath is null)
                {
                    pendingFinalOutputPath = _options.BuildCompletedAutoOutputModelPath(DateTimeOffset.Now, trainSamples.Count, bestSavedDecisionThreshold);
                }

                capturedException = ExceptionDispatchInfo.Capture(ex);
            }
            finally
            {
                optimizer?.Dispose();
            }
        }

        if (capturedException is null)
        {
            if (bestModelSaved && pendingFinalOutputPath is not null)
            {
                PromoteAutoNamedOutput(modelOutputPath, pendingFinalOutputPath);
                modelOutputPath = pendingFinalOutputPath;
                bestModelPromoted = true;
                Console.WriteLine($"Final auto-named model: {modelOutputPath}");
            }

            PrintTimingSummary(
                trainPreprocessStopwatch.Elapsed,
                validationPreprocessStopwatch.Elapsed,
                epochTimings,
                totalStopwatch.Elapsed);
            return;
        }

        if (bestModelSaved && !bestModelPromoted && pendingFinalOutputPath is not null)
        {
            bestModelPromoted = !string.IsNullOrWhiteSpace(
                TryPromoteInterruptedBestModel(modelOutputPath, pendingFinalOutputPath));
        }

        capturedException.Throw();
    }

    public void RequestCancellation()
    {
        _cancellationRequested = true;
    }

    private EpochMetrics RunEpoch(
        PopularityModel model,
        IReadOnlyList<LabeledImageSample> samples,
        optim.Optimizer? optimizer,
        IReadOnlyList<PopularityModel.LayerwiseParameterGroup> layerwiseParameterGroups,
        bool isTraining,
        int epoch,
        int trainableBackboneStageCount,
        Func<string?>? progressMemoryProvider)
    {
        var phaseStopwatch = Stopwatch.StartNew();

        if (isTraining)
        {
            model.train();

            model.ConfigureBackboneTrainability(trainableBackboneStageCount);
        }
        else
        {
            model.eval();
        }

        double totalLoss = 0;
        long totalCorrect = 0;
        long totalCount = 0;
        double popularLoss = 0;
        long popularCorrect = 0;
        long popularCount = 0;
        double unpopularLoss = 0;
        long unpopularCorrect = 0;
        long unpopularCount = 0;
        double totalWeight = 0;
        List<float>? validationProbabilities = isTraining ? null : new List<float>(samples.Count);
        List<float>? validationLabels = isTraining ? null : new List<float>(samples.Count);

        using var noGrad = isTraining ? null : torch.no_grad();
        var plannedBatches = Math.Max(1, (int)Math.Ceiling(samples.Count / (double)_options.BatchSize));
        var phaseLabel = isTraining ? $"Train E{epoch}/{_options.Epochs}" : $"Valid E{epoch}/{_options.Epochs}";
        using var progress = new ConsoleProgressBar(
            phaseLabel,
            plannedBatches,
            clearOnComplete: true,
            dynamicStatusProvider: progressMemoryProvider);

        var cache = isTraining ? _trainPreprocessedImageCache : _validationPreprocessedImageCache;
        var processedBatches = 0;
        foreach (var batch in CreateBatches(samples, _options.BatchSize, shuffle: isTraining, cache: cache))
        {
            ThrowIfCancellationRequested();

            try
            {
                using var inputs = batch.Inputs.to(_device);
                using var labels = batch.Labels.to(_device);
                using var sampleWeights = batch.SampleWeights.to(_device);

                if (isTraining)
                {
                    optimizer!.zero_grad();
                }

                using var logits = model.forward(inputs);
                using var perSampleLoss = nn.functional.softplus(logits) - (logits * labels);
                using var loss = (perSampleLoss * sampleWeights).sum() / sampleWeights.sum();

                if (isTraining)
                {
                    loss.backward();
                    ApplyLayerwiseLearningRateScales(layerwiseParameterGroups);
                    optimizer!.step();
                }

                var batchSize = labels.shape[0];
                totalLoss += (perSampleLoss * sampleWeights).sum().item<float>();
                totalWeight += sampleWeights.sum().item<float>();

                using var probabilities = torch.sigmoid(logits);
                using var predictions = torch.ge(probabilities, _options.DecisionThreshold);
                using var expected = torch.ge(labels, 0.5);
                using var correct = torch.eq(predictions, expected);

                totalCorrect += correct.sum().item<long>();
                totalCount += batchSize;
                processedBatches++;

                if (!isTraining)
                {
                    validationProbabilities!.AddRange(probabilities.to(CPU).data<float>().ToArray());
                    validationLabels!.AddRange(labels.to(CPU).data<float>().ToArray());

                    using var popularMask = torch.ge(labels, 0.5);
                    using var unpopularMask = torch.lt(labels, 0.5);

                    var batchPopularCount = popularMask.sum().item<long>();
                    if (batchPopularCount > 0)
                    {
                        using var popularLosses = perSampleLoss.masked_select(popularMask);
                        using var popularCorrectTensor = correct.masked_select(popularMask);
                        popularLoss += popularLosses.sum().item<float>();
                        popularCorrect += popularCorrectTensor.sum().item<long>();
                        popularCount += batchPopularCount;
                    }

                    var batchUnpopularCount = unpopularMask.sum().item<long>();
                    if (batchUnpopularCount > 0)
                    {
                        using var unpopularLosses = perSampleLoss.masked_select(unpopularMask);
                        using var unpopularCorrectTensor = correct.masked_select(unpopularMask);
                        unpopularLoss += unpopularLosses.sum().item<float>();
                        unpopularCorrect += unpopularCorrectTensor.sum().item<long>();
                        unpopularCount += batchUnpopularCount;
                    }
                }

                var avgLoss = totalCount == 0 ? 0 : totalLoss / totalCount;
                if (isTraining && totalWeight > 0)
                {
                    avgLoss = totalLoss / totalWeight;
                }
                var avgAccuracy = totalCount == 0 ? 0 : totalCorrect / (double)totalCount;
                progress.Report(processedBatches, $"loss={avgLoss:F4} acc={avgAccuracy:P2}");
            }
            finally
            {
                batch.Inputs.Dispose();
                batch.Labels.Dispose();
                batch.SampleWeights.Dispose();
            }
        }

        var metrics = new EpochMetrics
        {
            Loss = totalCount == 0
                ? double.MaxValue
                : isTraining && totalWeight > 0
                    ? totalLoss / totalWeight
                    : totalLoss / totalCount,
            Accuracy = totalCount == 0 ? 0 : totalCorrect / (double)totalCount,
            Duration = phaseStopwatch.Elapsed,
            PopularLoss = popularCount == 0 ? 0 : popularLoss / popularCount,
            PopularAccuracy = popularCount == 0 ? 0 : popularCorrect / (double)popularCount,
            PopularCount = popularCount,
            UnpopularLoss = unpopularCount == 0 ? 0 : unpopularLoss / unpopularCount,
            UnpopularAccuracy = unpopularCount == 0 ? 0 : unpopularCorrect / (double)unpopularCount,
            UnpopularCount = unpopularCount,
            DecisionThresholdUsed = _options.DecisionThreshold,
            BalancedLoss = popularCount == 0 || unpopularCount == 0
                ? totalCount == 0
                    ? 0
                    : isTraining && totalWeight > 0
                        ? totalLoss / totalWeight
                        : totalLoss / totalCount
                : ((popularLoss / popularCount) + (unpopularLoss / unpopularCount)) / 2d,
            BalancedAccuracy = popularCount == 0 || unpopularCount == 0
                ? totalCount == 0 ? 0 : totalCorrect / (double)totalCount
                : ((popularCorrect / (double)popularCount) + (unpopularCorrect / (double)unpopularCount)) / 2d
        };

        if (!isTraining && validationProbabilities is not null && validationLabels is not null && validationProbabilities.Count > 0)
        {
            var thresholdMetrics = FindBestValidationThreshold(validationProbabilities, validationLabels, _options.DecisionThreshold);
            metrics = metrics with
            {
                Accuracy = thresholdMetrics.Accuracy,
                PopularAccuracy = thresholdMetrics.PopularAccuracy,
                UnpopularAccuracy = thresholdMetrics.UnpopularAccuracy,
                DecisionThresholdUsed = thresholdMetrics.DecisionThreshold,
                BalancedAccuracy = thresholdMetrics.BalancedAccuracy
            };
        }

        progress.Complete($"loss={metrics.Loss:F4} acc={metrics.Accuracy:P2}");

        return metrics;
    }

    private IEnumerable<TrainingBatch> CreateBatches(
        IReadOnlyList<LabeledImageSample> samples,
        int batchSize,
        bool shuffle,
        PreprocessedImageCache cache)
    {
        var ordered = shuffle
            ? BuildBalancedTrainingOrder(samples, batchSize)
            : samples.ToList();

        var imageTensorSize = 3 * _options.TrainImageSize * _options.TrainImageSize;

        for (var start = 0; start < ordered.Count; start += batchSize)
        {
            ThrowIfCancellationRequested();

            var count = Math.Min(batchSize, ordered.Count - start);
            var features = new float[count * imageTensorSize];
            var labels = new float[count];
            var weights = new float[count];

            var valid = 0;

            for (var i = 0; i < count; i++)
            {
                var sample = ordered[start + i];

                try
                {
                    var chw = cache.LoadChw(sample.ImagePath);
                    Array.Copy(chw, 0, features, valid * imageTensorSize, imageTensorSize);
                    labels[valid] = sample.Label;
                    weights[valid] = sample.TrainingWeight;
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
                Array.Resize(ref weights, valid);
            }

            var inputTensor = torch.tensor(features, dtype: ScalarType.Float32)
                .reshape(valid, 3, _options.TrainImageSize, _options.TrainImageSize);
            var labelTensor = torch.tensor(labels, dtype: ScalarType.Float32)
                .reshape(valid, 1);
            var weightTensor = torch.tensor(weights, dtype: ScalarType.Float32)
                .reshape(valid, 1);

            yield return new TrainingBatch(inputTensor, labelTensor, weightTensor);
        }
    }

    private (IReadOnlyList<LabeledImageSample> Train, IReadOnlyList<LabeledImageSample> Validation) LoadDatasets()
    {
        if (_options.ValidationFileNames.Count == 0)
        {
            var allSamples = LoadSamplesForRandomSplit();
            return SplitStratified(allSamples, _options.ValidationSplit);
        }

        return LoadSamplesFromExplicitValidationFiles(_options.ValidationFileNames);
    }

    private IReadOnlyList<LabeledImageSample> LoadSamplesForRandomSplit()
    {
        var popular = EnumerateImages(_options.PopularDirectory)
            .Select(path => CreateSample(path, 1f))
            .ToList();

        var unpopular = EnumerateImages(_options.UnpopularDirectory)
            .Select(path => CreateSample(path, 0f))
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

        if (_options.EnableGroupAwareTraining)
        {
            popular = ApplyTrainingWeights(popular);
            unpopular = ApplyTrainingWeights(unpopular);
        }

        if (popular.Count == 0 || unpopular.Count == 0)
        {
            throw new InvalidOperationException("Both popular and unpopular directories must contain images.");
        }

        Console.WriteLine($"Popular images: {popular.Count} (groups: {CountDistinctGroups(popular)})");
        Console.WriteLine($"Unpopular images: {unpopular.Count} (groups: {CountDistinctGroups(unpopular)})");
        Console.WriteLine($"Validation selection: random stratified split ({_options.ValidationSplit:P0})");

        return [.. popular, .. unpopular];
    }

    private (IReadOnlyList<LabeledImageSample> Train, IReadOnlyList<LabeledImageSample> Validation) LoadSamplesFromExplicitValidationFiles(IReadOnlyList<string> validationFileNames)
    {
        var popularValidationFiles = ResolveExistingValidationFiles(_options.PopularDirectory, validationFileNames);
        var unpopularValidationFiles = ResolveExistingValidationFiles(_options.UnpopularDirectory, validationFileNames);

        var trainPopular = EnumerateImages(_options.PopularDirectory)
            .Select(path => CreateSample(path, 1f))
            .ToList();
        var trainUnpopular = EnumerateImages(_options.UnpopularDirectory)
            .Select(path => CreateSample(path, 0f))
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

        var validationPopular = EnumerateValidationImagesFromFiles(popularValidationFiles)
            .Select(path => CreateSample(path, 1f))
            .ToList();
        var validationUnpopular = EnumerateValidationImagesFromFiles(unpopularValidationFiles)
            .Select(path => CreateSample(path, 0f))
            .ToList();

        var allValidationPaths = validationPopular
            .Select(sample => sample.ImagePath)
            .Concat(validationUnpopular.Select(sample => sample.ImagePath))
            .ToHashSet(StringComparer.OrdinalIgnoreCase);
        trainPopular = trainPopular
            .Where(sample => !allValidationPaths.Contains(sample.ImagePath))
            .ToList();
        trainUnpopular = trainUnpopular
            .Where(sample => !allValidationPaths.Contains(sample.ImagePath))
            .ToList();

        if (_options.EnableGroupAwareTraining)
        {
            var validationPopularGroups = validationPopular.Select(sample => sample.EffectiveGroupKey).ToHashSet(StringComparer.OrdinalIgnoreCase);
            var validationUnpopularGroups = validationUnpopular.Select(sample => sample.EffectiveGroupKey).ToHashSet(StringComparer.OrdinalIgnoreCase);
            trainPopular = trainPopular
                .Where(sample => !validationPopularGroups.Contains(sample.EffectiveGroupKey))
                .ToList();
            trainUnpopular = trainUnpopular
                .Where(sample => !validationUnpopularGroups.Contains(sample.EffectiveGroupKey))
                .ToList();
            trainPopular = ApplyTrainingWeights(trainPopular);
            trainUnpopular = ApplyTrainingWeights(trainUnpopular);
        }

        if (trainPopular.Count == 0 || trainUnpopular.Count == 0)
        {
            throw new InvalidOperationException("Explicit validation files left no training images in one or both classes.");
        }

        var validationCount = validationPopular.Count + validationUnpopular.Count;
        var totalCount = trainPopular.Count + trainUnpopular.Count + validationCount;
        var minimumValidationCount = (int)Math.Ceiling(totalCount * _options.ValidationSplit);
        if (validationCount < minimumValidationCount)
        {
            throw new InvalidOperationException(
                $"Explicit validation set is too small: found {validationCount} image(s), but at least {minimumValidationCount} are required by validation-split={_options.ValidationSplit.ToString("0.####", System.Globalization.CultureInfo.InvariantCulture)}.");
        }

        Console.WriteLine($"Popular train images: {trainPopular.Count} (groups: {CountDistinctGroups(trainPopular)})");
        Console.WriteLine($"Unpopular train images: {trainUnpopular.Count} (groups: {CountDistinctGroups(trainUnpopular)})");
        Console.WriteLine($"Popular validation images: {validationPopular.Count} (groups: {CountDistinctGroups(validationPopular)})");
        Console.WriteLine($"Unpopular validation images: {validationUnpopular.Count} (groups: {CountDistinctGroups(validationUnpopular)})");
        Console.WriteLine($"Validation selection: explicit file names [{string.Join(", ", validationFileNames)}]");
        Console.WriteLine($"Matched popular validation files: {FormatPathList(popularValidationFiles)}");
        Console.WriteLine($"Matched unpopular validation files: {FormatPathList(unpopularValidationFiles)}");

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

    private static IEnumerable<string> EnumerateValidationImagesFromFiles(IReadOnlyList<string> files)
    {
        return files
            .SelectMany(EnumerateValidationImagesFromFile)
            .Distinct(StringComparer.OrdinalIgnoreCase);
    }

    private static IEnumerable<string> EnumerateValidationImagesFromFile(string path)
    {
        if (SupportedImageFiles.IsSupported(path))
        {
            yield return Path.GetFullPath(path);
            yield break;
        }

        if (!string.Equals(Path.GetExtension(path), ".txt", StringComparison.OrdinalIgnoreCase))
        {
            yield break;
        }

        foreach (var referencedImagePath in EnumerateTextReferencedImages(path, excludedDirectories: null))
        {
            yield return referencedImagePath;
        }
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

        return EnumerateImagesAndTextReferences(directory, normalizedExcluded);
    }

    private static IEnumerable<string> EnumerateImagesStatic(string directory)
    {
        if (!Directory.Exists(directory))
        {
            return Array.Empty<string>();
        }

        return EnumerateImagesAndTextReferences(directory, excludedDirectories: null);
    }

    private static IEnumerable<string> EnumerateImagesAndTextReferences(string directory, IReadOnlyList<string>? excludedDirectories)
    {
        var results = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        foreach (var path in Directory.EnumerateFiles(directory, "*", SearchOption.AllDirectories))
        {
            if (excludedDirectories is not null && IsUnderAnyDirectory(path, excludedDirectories))
            {
                continue;
            }

            if (SupportedImageFiles.IsSupported(path))
            {
                results.Add(Path.GetFullPath(path));
                continue;
            }

            if (!string.Equals(Path.GetExtension(path), ".txt", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            foreach (var referencedImagePath in EnumerateTextReferencedImages(path, excludedDirectories))
            {
                results.Add(referencedImagePath);
            }
        }

        return results;
    }

    private static IEnumerable<string> EnumerateTextReferencedImages(string textFilePath, IReadOnlyList<string>? excludedDirectories)
    {
        var textFileDirectory = Path.GetDirectoryName(textFilePath) ?? Directory.GetCurrentDirectory();
        var lineNumber = 0;

        foreach (var rawLine in File.ReadLines(textFilePath))
        {
            lineNumber++;
            var referencedImagePath = TryResolveReferencedImagePath(rawLine, textFileDirectory);
            if (referencedImagePath is null)
            {
                continue;
            }

            if (excludedDirectories is not null && IsUnderAnyDirectory(referencedImagePath, excludedDirectories))
            {
                continue;
            }

            if (!SupportedImageFiles.IsSupported(referencedImagePath))
            {
                Console.WriteLine($"Skip unsupported image reference in {textFilePath}:{lineNumber} -> {referencedImagePath}");
                continue;
            }

            if (!File.Exists(referencedImagePath))
            {
                Console.WriteLine($"Skip missing image reference in {textFilePath}:{lineNumber} -> {referencedImagePath}");
                continue;
            }

            yield return referencedImagePath;
        }
    }

    private static string? TryResolveReferencedImagePath(string rawLine, string textFileDirectory)
    {
        if (string.IsNullOrWhiteSpace(rawLine))
        {
            return null;
        }

        var trimmed = rawLine.Trim();
        if (trimmed.Length == 0 ||
            trimmed.StartsWith("#", StringComparison.Ordinal) ||
            trimmed.StartsWith("//", StringComparison.Ordinal))
        {
            return null;
        }

        trimmed = trimmed.Trim('"');
        if (trimmed.Length == 0)
        {
            return null;
        }

        var resolved = Path.IsPathRooted(trimmed)
            ? trimmed
            : Path.Combine(textFileDirectory, trimmed);

        return Path.GetFullPath(resolved);
    }

    private (IReadOnlyList<LabeledImageSample> Train, IReadOnlyList<LabeledImageSample> Validation) SplitStratified(
        IReadOnlyList<LabeledImageSample> all,
        double validationSplit)
    {
        if (_options.EnableGroupAwareTraining)
        {
            return SplitStratifiedByGroup(all, validationSplit);
        }

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

    private (IReadOnlyList<LabeledImageSample> Train, IReadOnlyList<LabeledImageSample> Validation) SplitStratifiedByGroup(
        IReadOnlyList<LabeledImageSample> all,
        double validationSplit)
    {
        var (trainPopular, validationPopular) = SplitClassByGroup(all.Where(x => x.Label > 0.5f).ToList(), validationSplit);
        var (trainUnpopular, validationUnpopular) = SplitClassByGroup(all.Where(x => x.Label < 0.5f).ToList(), validationSplit);

        var train = ApplyTrainingWeights(trainPopular)
            .Concat(ApplyTrainingWeights(trainUnpopular))
            .OrderBy(_ => _random.Next())
            .ToList();
        var validation = validationPopular
            .Concat(validationUnpopular)
            .OrderBy(_ => _random.Next())
            .ToList();

        return (train, validation);
    }

    private (List<LabeledImageSample> Train, List<LabeledImageSample> Validation) SplitClassByGroup(
        IReadOnlyList<LabeledImageSample> samples,
        double validationSplit)
    {
        var grouped = samples
            .GroupBy(sample => sample.EffectiveGroupKey, StringComparer.OrdinalIgnoreCase)
            .OrderBy(_ => _random.Next())
            .ToList();

        var targetValidationCount = Math.Max(1, (int)Math.Round(samples.Count * validationSplit));
        var validationGroups = new List<IGrouping<string, LabeledImageSample>>();
        var trainGroups = new List<IGrouping<string, LabeledImageSample>>();
        var currentValidationCount = 0;

        foreach (var group in grouped)
        {
            if (currentValidationCount < targetValidationCount)
            {
                validationGroups.Add(group);
                currentValidationCount += group.Count();
            }
            else
            {
                trainGroups.Add(group);
            }
        }

        if (trainGroups.Count == 0 && validationGroups.Count > 1)
        {
            trainGroups.Add(validationGroups[^1]);
            currentValidationCount -= validationGroups[^1].Count();
            validationGroups.RemoveAt(validationGroups.Count - 1);
        }

        var validation = validationGroups.SelectMany(group => group).ToList();
        var train = trainGroups.SelectMany(group => group).ToList();
        return (train, validation);
    }

    private List<LabeledImageSample> ApplyTrainingWeights(IReadOnlyList<LabeledImageSample> samples)
    {
        if (!_options.EnableGroupAwareTraining || samples.Count == 0)
        {
            return samples.ToList();
        }

        var groupSizes = samples
            .GroupBy(sample => sample.EffectiveGroupKey, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(group => group.Key, group => group.Count(), StringComparer.OrdinalIgnoreCase);

        return samples
            .Select(sample =>
            {
                var groupSize = groupSizes[sample.EffectiveGroupKey];
                var weight = (float)(1d / Math.Pow(groupSize, GroupWeightExponent));
                return sample with { TrainingWeight = weight };
            })
            .ToList();
    }

    private static LabeledImageSample CreateSample(string imagePath, float label)
    {
        var normalizedImagePath = Path.GetFullPath(imagePath);
        var parsedGroupId = TryParseLeadingGroupId(normalizedImagePath);
        var effectiveGroupKey = parsedGroupId is not null
            ? $"g:{parsedGroupId}"
            : $"f:{normalizedImagePath}";
        return new LabeledImageSample(normalizedImagePath, label, effectiveGroupKey, parsedGroupId, 1f);
    }

    private static string? TryParseLeadingGroupId(string imagePath)
    {
        var fileName = Path.GetFileNameWithoutExtension(imagePath);
        if (string.IsNullOrWhiteSpace(fileName))
        {
            return null;
        }

        var underscoreIndex = fileName.IndexOf('_');
        var candidate = underscoreIndex >= 0
            ? fileName[..underscoreIndex]
            : fileName;

        if (candidate.Length == 0 || !candidate.All(char.IsDigit))
        {
            return null;
        }

        return candidate;
    }

    private static int CountDistinctGroups(IEnumerable<LabeledImageSample> samples)
    {
        return samples
            .Select(sample => sample.EffectiveGroupKey)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Count();
    }

    private void ShuffleInPlace<T>(IList<T> values)
    {
        for (var i = values.Count - 1; i > 0; i--)
        {
            var j = _random.Next(i + 1);
            (values[i], values[j]) = (values[j], values[i]);
        }
    }

    private List<LabeledImageSample> BuildBalancedTrainingOrder(IReadOnlyList<LabeledImageSample> samples, int batchSize)
    {
        var popular = samples.Where(sample => sample.Label > 0.5f).ToList();
        var unpopular = samples.Where(sample => sample.Label < 0.5f).ToList();

        if (popular.Count == 0 || unpopular.Count == 0)
        {
            var fallback = samples.ToList();
            ShuffleInPlace(fallback);
            return fallback;
        }

        ShuffleInPlace(popular);
        ShuffleInPlace(unpopular);

        var popularPerBatch = (batchSize + 1) / 2;
        var unpopularPerBatch = batchSize / 2;
        var batches = (int)Math.Ceiling((Math.Max(popular.Count, unpopular.Count) * 2d) / batchSize);
        var ordered = new List<LabeledImageSample>(batches * batchSize);
        var popularIndex = 0;
        var unpopularIndex = 0;

        for (var batch = 0; batch < batches; batch++)
        {
            var currentBatch = new List<LabeledImageSample>(batchSize);
            Dictionary<string, int>? batchGroupCounts = _options.EnableGroupAwareTraining
                ? new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase)
                : null;
            AddCycledSamples(popular, popularPerBatch, currentBatch, ref popularIndex, batchGroupCounts);
            AddCycledSamples(unpopular, unpopularPerBatch, currentBatch, ref unpopularIndex, batchGroupCounts);
            ShuffleInPlace(currentBatch);
            ordered.AddRange(currentBatch);
        }

        return ordered;
    }

    private void AddCycledSamples(
        IList<LabeledImageSample> source,
        int count,
        ICollection<LabeledImageSample> destination,
        ref int index,
        IDictionary<string, int>? batchGroupCounts)
    {
        if (source.Count == 0 || count <= 0)
        {
            return;
        }

        var added = 0;
        var attempts = 0;
        var maxAttempts = Math.Max(source.Count * 3, count * 4);

        while (added < count && attempts < maxAttempts)
        {
            attempts++;

            if (index >= source.Count)
            {
                ShuffleInPlace(source);
                index = 0;
            }

            var sample = source[index];
            index++;

            if (batchGroupCounts is not null)
            {
                batchGroupCounts.TryGetValue(sample.EffectiveGroupKey, out var existingCount);
                if (existingCount >= MaxSameGroupPerTrainingBatch)
                {
                    continue;
                }

                batchGroupCounts[sample.EffectiveGroupKey] = existingCount + 1;
            }

            destination.Add(sample);
            added++;
        }
    }

    private optim.Optimizer CreateOptimizer(IReadOnlyList<PopularityModel.LayerwiseParameterGroup> parameterGroups, double learningRate)
    {
        var uniqueParameters = new HashSet<TorchSharp.Modules.Parameter>(ReferenceEqualityComparer.Instance);
        foreach (var parameter in parameterGroups.SelectMany(group => group.Parameters))
        {
            uniqueParameters.Add(parameter);
        }

        var trainableParameters = uniqueParameters.ToArray();

        if (trainableParameters.Length == 0)
        {
            throw new InvalidOperationException("No trainable parameters found for optimizer.");
        }

        return optim.AdamW(trainableParameters, lr: learningRate, weight_decay: _options.WeightDecay);
    }

    private void ApplyLayerwiseLearningRateScales(IReadOnlyList<PopularityModel.LayerwiseParameterGroup> parameterGroups)
    {
        foreach (var parameterGroup in parameterGroups)
        {
            if (Math.Abs(parameterGroup.LearningRateScale - 1.0) < 1e-9)
            {
                continue;
            }

            foreach (var parameter in parameterGroup.Parameters)
            {
                using var gradient = parameter.grad;
                if (gradient is null)
                {
                    continue;
                }

                gradient.mul_(parameterGroup.LearningRateScale);
            }
        }
    }

    private static bool IsBetterBestModelCandidate(EpochMetrics candidateMetrics, LossPriorityState currentBest)
    {
        const double tolerance = 1e-9;
        if (!currentBest.HasValue)
        {
            return true;
        }

        var candidateMeetsPopularTarget = candidateMetrics.PopularLoss < PopularLossTarget;

        if (candidateMeetsPopularTarget != currentBest.MeetsPopularTarget)
        {
            return candidateMeetsPopularTarget;
        }

        if (candidateMeetsPopularTarget)
        {
            if (candidateMetrics.UnpopularLoss < currentBest.UnpopularLoss - tolerance)
            {
                return true;
            }

            if (Math.Abs(candidateMetrics.UnpopularLoss - currentBest.UnpopularLoss) <= tolerance &&
                candidateMetrics.Loss < currentBest.TotalLoss - tolerance)
            {
                return true;
            }

            if (Math.Abs(candidateMetrics.UnpopularLoss - currentBest.UnpopularLoss) <= tolerance &&
                Math.Abs(candidateMetrics.Loss - currentBest.TotalLoss) <= tolerance &&
                candidateMetrics.PopularLoss < currentBest.PopularLoss - tolerance)
            {
                return true;
            }

            return false;
        }

        if (candidateMetrics.PopularLoss < currentBest.PopularLoss - tolerance)
        {
            return true;
        }

        if (Math.Abs(candidateMetrics.PopularLoss - currentBest.PopularLoss) <= tolerance &&
            candidateMetrics.Loss < currentBest.TotalLoss - tolerance)
        {
            return true;
        }

        if (Math.Abs(candidateMetrics.PopularLoss - currentBest.PopularLoss) <= tolerance &&
            Math.Abs(candidateMetrics.Loss - currentBest.TotalLoss) <= tolerance &&
            candidateMetrics.UnpopularLoss < currentBest.UnpopularLoss - tolerance)
        {
            return true;
        }

        return false;
    }

    private static bool HasMeaningfulEarlyStoppingImprovement(
        EpochMetrics candidateMetrics,
        LossPriorityState currentBest,
        double minDelta)
    {
        if (!currentBest.HasValue)
        {
            return true;
        }

        var candidateMeetsPopularTarget = candidateMetrics.PopularLoss < PopularLossTarget;

        if (!currentBest.MeetsPopularTarget)
        {
            if (candidateMeetsPopularTarget)
            {
                return true;
            }

            return candidateMetrics.PopularLoss < currentBest.PopularLoss - minDelta;
        }

        if (!candidateMeetsPopularTarget)
        {
            return false;
        }

        if (candidateMetrics.UnpopularLoss < currentBest.UnpopularLoss - minDelta)
        {
            return true;
        }

        return Math.Abs(candidateMetrics.UnpopularLoss - currentBest.UnpopularLoss) <= minDelta &&
               candidateMetrics.Loss < currentBest.TotalLoss - minDelta;
    }

    private static ThresholdMetrics FindBestValidationThreshold(
        IReadOnlyList<float> probabilities,
        IReadOnlyList<float> labels,
        double defaultThreshold)
    {
        var best = EvaluateThreshold(probabilities, labels, defaultThreshold);

        for (var threshold = ThresholdScanMinimum; threshold <= ThresholdScanMaximum + 1e-9; threshold += ThresholdScanStep)
        {
            var metrics = EvaluateThreshold(probabilities, labels, threshold);
            if (metrics.BalancedAccuracy > best.BalancedAccuracy + 1e-9)
            {
                best = metrics;
                continue;
            }

            if (Math.Abs(metrics.BalancedAccuracy - best.BalancedAccuracy) <= 1e-9 &&
                metrics.Accuracy > best.Accuracy + 1e-9)
            {
                best = metrics;
                continue;
            }

            if (Math.Abs(metrics.BalancedAccuracy - best.BalancedAccuracy) <= 1e-9 &&
                Math.Abs(metrics.Accuracy - best.Accuracy) <= 1e-9 &&
                Math.Abs(threshold - defaultThreshold) <
                Math.Abs(best.DecisionThreshold - defaultThreshold))
            {
                best = metrics;
            }
        }

        return best;
    }

    private static ThresholdMetrics EvaluateThreshold(
        IReadOnlyList<float> probabilities,
        IReadOnlyList<float> labels,
        double threshold)
    {
        long totalCorrect = 0;
        long popularCorrect = 0;
        long popularCount = 0;
        long unpopularCorrect = 0;
        long unpopularCount = 0;

        for (var i = 0; i < probabilities.Count; i++)
        {
            var label = labels[i] >= 0.5f;
            var prediction = probabilities[i] >= threshold;
            if (prediction == label)
            {
                totalCorrect++;
            }

            if (label)
            {
                popularCount++;
                if (prediction)
                {
                    popularCorrect++;
                }
            }
            else
            {
                unpopularCount++;
                if (!prediction)
                {
                    unpopularCorrect++;
                }
            }
        }

        var accuracy = probabilities.Count == 0 ? 0 : totalCorrect / (double)probabilities.Count;
        var popularAccuracy = popularCount == 0 ? 0 : popularCorrect / (double)popularCount;
        var unpopularAccuracy = unpopularCount == 0 ? 0 : unpopularCorrect / (double)unpopularCount;
        var balancedAccuracy = popularCount == 0 || unpopularCount == 0
            ? accuracy
            : (popularAccuracy + unpopularAccuracy) / 2d;

        return new ThresholdMetrics(threshold, accuracy, popularAccuracy, unpopularAccuracy, balancedAccuracy);
    }

    private int GetTrainableBackboneStageCountForEpoch(PopularityModel model, int epoch)
    {
        if (model.BackboneStageCount == 0)
        {
            return _options.FreezeBackboneEpochs > 0 && epoch <= _options.FreezeBackboneEpochs
                ? 0
                : 1;
        }

        if (_options.FreezeBackboneEpochs <= 0)
        {
            return model.BackboneStageCount;
        }

        if (epoch <= _options.FreezeBackboneEpochs)
        {
            return 0;
        }

        var progressiveEpoch = epoch - _options.FreezeBackboneEpochs;
        return Math.Min(model.BackboneStageCount, progressiveEpoch);
    }

    private static string FormatStageSummary(IReadOnlyList<string> stageNames)
    {
        return stageNames.Count == 0
            ? "(head only)"
            : string.Join(", ", stageNames);
    }

    private static string FormatLayerwiseScaleSummary(IReadOnlyList<PopularityModel.LayerwiseParameterGroup> parameterGroups)
    {
        return string.Join(
            ", ",
            parameterGroups.Select(group => $"{group.Name}={group.LearningRateScale.ToString("0.##", CultureInfo.InvariantCulture)}"));
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

    private static string FormatLearningRate(double learningRate)
    {
        return learningRate.ToString("0.################", CultureInfo.InvariantCulture);
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

    private string? TryPromoteInterruptedBestModel(string autosaveModelPath, string finalOutputPath)
    {
        try
        {
            if (!File.Exists(autosaveModelPath))
            {
                return null;
            }

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

    private static IReadOnlyList<string> ResolveExistingValidationFiles(string classRoot, IReadOnlyList<string> validationFileNames)
    {
        if (!Directory.Exists(classRoot))
        {
            return Array.Empty<string>();
        }

        var requestedFileNames = validationFileNames.ToHashSet(StringComparer.OrdinalIgnoreCase);
        var matchedFiles = Directory
            .EnumerateFiles(classRoot, "*", SearchOption.AllDirectories)
            .Where(path => requestedFileNames.Contains(Path.GetFileName(path)))
            .Where(path => SupportedImageFiles.IsSupported(path) || string.Equals(Path.GetExtension(path), ".txt", StringComparison.OrdinalIgnoreCase))
            .Select(Path.GetFullPath)
            .ToArray();

        return matchedFiles
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }

    private static string FormatPathList(IReadOnlyList<string> paths)
    {
        return paths.Count == 0
            ? "(none)"
            : string.Join(", ", paths);
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

    private GpuMemoryStatusSampler? CreateGpuMemoryStatusSampler()
    {
        var deviceIndex = TryGetCudaDeviceIndex(_device);
        return new GpuMemoryStatusSampler(deviceIndex);
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

    private readonly record struct LabeledImageSample(
        string ImagePath,
        float Label,
        string EffectiveGroupKey,
        string? ParsedGroupId,
        float TrainingWeight);

    private readonly record struct TrainingBatch(Tensor Inputs, Tensor Labels, Tensor SampleWeights);

    private sealed record EpochMetrics
    {
        public double Loss { get; init; }

        public double Accuracy { get; init; }

        public TimeSpan Duration { get; init; }

        public double PopularLoss { get; init; }

        public double PopularAccuracy { get; init; }

        public long PopularCount { get; init; }

        public double UnpopularLoss { get; init; }

        public double UnpopularAccuracy { get; init; }

        public long UnpopularCount { get; init; }

        public double DecisionThresholdUsed { get; init; } = double.NaN;

        public double BalancedLoss { get; init; }

        public double BalancedAccuracy { get; init; }

        public bool HasClassBreakdown => PopularCount > 0 || UnpopularCount > 0;
    }

    private readonly record struct ThresholdMetrics(
        double DecisionThreshold,
        double Accuracy,
        double PopularAccuracy,
        double UnpopularAccuracy,
        double BalancedAccuracy);

    private readonly record struct LossPriorityState(
        double PopularLoss,
        double UnpopularLoss,
        double TotalLoss,
        bool MeetsPopularTarget,
        bool HasValue)
    {
        public static LossPriorityState Unset => new(double.MaxValue, double.MaxValue, double.MaxValue, false, false);

        public static LossPriorityState FromEpochMetrics(EpochMetrics metrics) =>
            new(
                metrics.PopularLoss,
                metrics.UnpopularLoss,
                metrics.Loss,
                metrics.PopularLoss < PopularLossTarget,
                true);
    }

    private readonly record struct EpochTiming(int Epoch, TimeSpan TrainDuration, TimeSpan ValidationDuration, TimeSpan TotalDuration);

    private sealed class GpuMemoryStatusSampler : IDisposable
    {
        private static readonly TimeSpan RefreshInterval = TimeSpan.FromSeconds(1);

        private readonly int? _deviceIndex;
        private readonly CancellationTokenSource _cts = new();
        private readonly Task _samplingTask;
        private volatile string? _latestStatus;
        private bool _disposed;

        public GpuMemoryStatusSampler(int? deviceIndex)
        {
            _deviceIndex = deviceIndex;
            _samplingTask = Task.Run(() => SampleLoopAsync(_cts.Token));
        }

        public string? GetStatus()
        {
            return _latestStatus;
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            _cts.Cancel();

            try
            {
                _samplingTask.Wait(TimeSpan.FromSeconds(2));
            }
            catch
            {
            }

            _cts.Dispose();
        }

        private async Task SampleLoopAsync(CancellationToken cancellationToken)
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                try
                {
                    var sampledStatus = QueryGpuMemoryStatus(_deviceIndex);
                    if (!string.IsNullOrWhiteSpace(sampledStatus))
                    {
                        _latestStatus = sampledStatus;
                    }
                }
                catch
                {
                }

                try
                {
                    await Task.Delay(RefreshInterval, cancellationToken).ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
            }
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
