using TorchSharp;
using static TorchSharp.torch;

namespace ImagePopularity.Core;

public sealed class ImagePopularityPredictor : IImagePopularityPredictor
{
    private readonly Device _device;
    private readonly int _imageSize;
    private readonly PopularityModel _model;
    private readonly InferencePreprocessCache? _preprocessCache;
    private readonly bool _enableTta;
    private bool _disposed;

    public ImagePopularityPredictor(string modelPath, ImagePopularityPredictorOptions? options = null)
    {
        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException($"Model not found: {modelPath}", modelPath);
        }

        options ??= new ImagePopularityPredictorOptions();
        var metadata = PopularityModelMetadata.TryLoad(modelPath);

        _device = TorchDeviceSelector.ResolveCuda();
        _imageSize = options.InferenceImageSize
            ?? metadata?.RecommendedInferenceImageSize
            ?? metadata?.TrainingImageSize
            ?? 320;

        if (_imageSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(options.InferenceImageSize), _imageSize, "Inference image size must be > 0.");
        }

        if (options.EnablePreprocessCache && string.IsNullOrWhiteSpace(options.PreprocessCacheDirectory))
        {
            throw new ArgumentException("Preprocess cache directory cannot be empty when EnablePreprocessCache is true.", nameof(options));
        }

        var backbone = options.Backbone ?? metadata?.Backbone ?? PopularityBackboneCatalog.DefaultBackbone;
        _enableTta = options.EnableTta;
        _preprocessCache = options.EnablePreprocessCache
            ? new InferencePreprocessCache(options.PreprocessCacheDirectory, _imageSize)
            : null;

        _model = new PopularityModel(new PopularityModelConfig
        {
            Backbone = backbone
        });
        _model.load(modelPath);
        _model.to(_device);
        _model.eval();
    }

    public float PredictProbability(string imagePath)
    {
        if (string.IsNullOrWhiteSpace(imagePath))
        {
            throw new ArgumentException("Image path cannot be empty.", nameof(imagePath));
        }

        return PredictProbabilities([imagePath], batchSize: 1)[0];
    }

    public IReadOnlyList<float> PredictProbabilities(IReadOnlyList<string> imagePaths, int batchSize = 64)
    {
        return PredictProbabilities(imagePaths, batchSize, progressCallback: null);
    }

    public IReadOnlyList<float> PredictProbabilities(
        IReadOnlyList<string> imagePaths,
        int batchSize,
        Action<int, int>? progressCallback)
    {
        EnsureNotDisposed();

        ArgumentNullException.ThrowIfNull(imagePaths);

        if (batchSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(batchSize), batchSize, "Batch size must be > 0.");
        }

        if (imagePaths.Count == 0)
        {
            return Array.Empty<float>();
        }

        var probabilities = new float[imagePaths.Count];
        var imageTensorSize = 3 * _imageSize * _imageSize;

        using var noGrad = torch.no_grad();

        for (var start = 0; start < imagePaths.Count; start += batchSize)
        {
            var currentBatch = Math.Min(batchSize, imagePaths.Count - start);
            var features = new float[currentBatch * imageTensorSize];
            float[]? ttaFeatures = _enableTta ? new float[currentBatch * imageTensorSize] : null;

            for (var i = 0; i < currentBatch; i++)
            {
                var imagePath = imagePaths[start + i];
                if (string.IsNullOrWhiteSpace(imagePath))
                {
                    throw new ArgumentException("Image path cannot be empty.", nameof(imagePaths));
                }

                var chw = _preprocessCache is not null
                    ? _preprocessCache.LoadChw(imagePath)
                    : ImageTensorFactory.LoadNormalizedChw(imagePath, _imageSize);
                Array.Copy(chw, 0, features, i * imageTensorSize, imageTensorSize);
                if (ttaFeatures is not null)
                {
                    WriteHorizontallyFlippedChw(chw, ttaFeatures, i * imageTensorSize, _imageSize);
                }
            }

            using var inputTensor = torch.tensor(features, dtype: ScalarType.Float32)
                .reshape(currentBatch, 3, _imageSize, _imageSize)
                .to(_device);

            using var logits = _model.forward(inputTensor);
            using var aggregatedLogits = _enableTta && ttaFeatures is not null
                ? AverageWithHorizontalFlipTtaLogits(logits, ttaFeatures, currentBatch)
                : logits.detach();
            using var batchProbabilities = torch.sigmoid(aggregatedLogits).to(CPU);

            for (var i = 0; i < currentBatch; i++)
            {
                probabilities[start + i] = batchProbabilities[i, 0].item<float>();
            }

            progressCallback?.Invoke(start + currentBatch, imagePaths.Count);
        }

        return probabilities;
    }

    private Tensor AverageWithHorizontalFlipTtaLogits(Tensor originalLogits, float[] ttaFeatures, int batchSize)
    {
        using var ttaInputTensor = torch.tensor(ttaFeatures, dtype: ScalarType.Float32)
            .reshape(batchSize, 3, _imageSize, _imageSize)
            .to(_device);
        using var ttaLogits = _model.forward(ttaInputTensor);
        return ((originalLogits + ttaLogits) / 2.0f).detach();
    }

    private static void WriteHorizontallyFlippedChw(float[] sourceChw, float[] destination, int destinationOffset, int imageSize)
    {
        var planeSize = imageSize * imageSize;
        for (var channel = 0; channel < 3; channel++)
        {
            var sourceChannelOffset = channel * planeSize;
            var destinationChannelOffset = destinationOffset + sourceChannelOffset;

            for (var y = 0; y < imageSize; y++)
            {
                var rowOffset = y * imageSize;
                for (var x = 0; x < imageSize; x++)
                {
                    destination[destinationChannelOffset + rowOffset + x] =
                        sourceChw[sourceChannelOffset + rowOffset + (imageSize - 1 - x)];
                }
            }
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _preprocessCache?.Dispose();
        _model.Dispose();
        _disposed = true;
    }

    private void EnsureNotDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(ImagePopularityPredictor));
        }
    }
}
