using System.Globalization;

namespace ImagePopularity.Core;

public sealed class ImagePopularityTrainingOptions
{
    public const double DefaultDecisionThreshold = 0.45;
    public const int DefaultEarlyStoppingPatience = 2;
    public const double DefaultEarlyStoppingMinDelta = 0.01;

    public required string PopularDirectory { get; init; }

    public required string UnpopularDirectory { get; init; }

    public string OutputModelPrefix { get; init; } = string.Empty;

    public string PreprocessCacheDirectory { get; init; } = Path.Combine("models", "preprocess-cache");

    public int TrainImageSize { get; init; } = 320;

    public int Epochs { get; init; } = 20;

    public int BatchSize { get; init; } = 128;

    public double LearningRate { get; init; } = 3e-4;

    public double FineTuneLearningRate { get; init; } = 5e-5;

    public double WeightDecay { get; init; } = 1e-4;

    public IReadOnlyList<string> ValidationDirectories { get; init; } = Array.Empty<string>();

    public double ValidationSplit { get; init; } = 0.1;

    public int Seed { get; init; } = 42;

    public int MaxSamplesPerClass { get; init; }

    public string Backbone { get; init; } = PopularityBackboneCatalog.DefaultBackbone;

    public string? PretrainedWeightsFile { get; init; }

    public int FreezeBackboneEpochs { get; init; } = 3;

    public bool EnableAugmentation { get; init; } = true;

    public double HorizontalFlipProbability { get; init; } = 0.5;

    public double MaxRotationDegrees { get; init; } = 12;

    public double BrightnessJitter { get; init; } = 0.15;

    public double ContrastJitter { get; init; } = 0.15;

    public double SaturationJitter { get; init; } = 0.15;

    public double MinRandomCropScale { get; init; } = 0.85;

    public double DecisionThreshold { get; init; } = DefaultDecisionThreshold;

    public int EarlyStoppingPatience { get; init; } = DefaultEarlyStoppingPatience;

    public double EarlyStoppingMinDelta { get; init; } = DefaultEarlyStoppingMinDelta;

    public string BuildInProgressOutputModelPath(int trainSampleCount)
    {
        return Path.Combine("models", BuildGeneratedModelFileName(trainSampleCount, completedAtLocal: null));
    }

    public string BuildCompletedAutoOutputModelPath(DateTimeOffset completedAtLocal, int trainSampleCount)
    {
        return Path.Combine("models", BuildGeneratedModelFileName(trainSampleCount, completedAtLocal));
    }

    private string BuildGeneratedModelFileName(int trainSampleCount, DateTimeOffset? completedAtLocal)
    {
        var augmentationTag = EnableAugmentation ? "a1" : "a0";
        var thresholdTag = BuildThresholdTag();
        var timestampSuffix = completedAtLocal is null
            ? string.Empty
            : $"_{completedAtLocal.Value.ToString("MMddHHmm", CultureInfo.InvariantCulture)}";
        return $"{OutputModelPrefix}{trainSampleCount}_{TrainImageSize}_{augmentationTag}_{thresholdTag}_e{Epochs}b{BatchSize}s{Seed}{timestampSuffix}.pt";
    }

    private string BuildThresholdTag()
    {
        var numeric = DecisionThreshold
            .ToString("0.##", CultureInfo.InvariantCulture)
            .Replace(".", string.Empty, StringComparison.Ordinal);

        return $"t{numeric}";
    }
}
