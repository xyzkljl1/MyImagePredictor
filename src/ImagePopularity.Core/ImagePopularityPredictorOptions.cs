namespace ImagePopularity.Core;

public sealed class ImagePopularityPredictorOptions
{
    public int? InferenceImageSize { get; init; }

    public string? Backbone { get; init; }

    public bool EnablePreprocessCache { get; init; }

    public string PreprocessCacheDirectory { get; init; } = Path.Combine("models", "inference-cache");

    public bool EnableTta { get; init; } = true;
}
