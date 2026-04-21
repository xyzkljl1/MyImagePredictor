namespace ImagePopularity.Core;

public sealed class ImageAugmentationOptions
{
    public bool Enabled { get; init; } = true;

    public float HorizontalFlipProbability { get; init; } = 0.5f;

    public float MaxRotationDegrees { get; init; } = 12f;

    public float BrightnessJitter { get; init; } = 0.15f;

    public float ContrastJitter { get; init; } = 0.15f;

    public float SaturationJitter { get; init; } = 0.15f;

    public float MinRandomCropScale { get; init; } = 0.85f;
}
