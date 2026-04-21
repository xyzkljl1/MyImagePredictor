using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Formats.Gif;

namespace ImagePopularity.Core;

public static class ImageTensorFactory
{
    private static readonly float[] Mean = [0.485f, 0.456f, 0.406f];
    private static readonly float[] Std = [0.229f, 0.224f, 0.225f];

    public static float[] LoadNormalizedChw(string imagePath, int imageSize)
    {
        return LoadNormalizedChw(imagePath, imageSize, augmentation: null, random: null);
    }

    public static float[] LoadNormalizedChw(
        string imagePath,
        int imageSize,
        ImageAugmentationOptions? augmentation,
        Random? random)
    {
        if (!File.Exists(imagePath))
        {
            throw new FileNotFoundException($"Image not found: {imagePath}", imagePath);
        }

        if (imageSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(imageSize), imageSize, "Image size must be > 0.");
        }

        using var image = LoadSingleFrameImage(imagePath);

        image.Mutate(x => x.AutoOrient());

        if (augmentation is { Enabled: true } && random is not null)
        {
            ApplyAugmentation(image, augmentation, random);
        }

        image.Mutate(x => x.Resize(new ResizeOptions
        {
            Size = new SixLabors.ImageSharp.Size(imageSize, imageSize),
            Mode = ResizeMode.Pad,
            Position = AnchorPositionMode.Center,
            Sampler = KnownResamplers.Bicubic
        }));

        var channelSize = imageSize * imageSize;
        var chw = new float[3 * channelSize];

        image.ProcessPixelRows(accessor =>
        {
            for (var y = 0; y < imageSize; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (var x = 0; x < imageSize; x++)
                {
                    var pixel = row[x];
                    var idx = y * imageSize + x;

                    chw[idx] = Normalize(pixel.R / 255f, Mean[0], Std[0]);
                    chw[channelSize + idx] = Normalize(pixel.G / 255f, Mean[1], Std[1]);
                    chw[(2 * channelSize) + idx] = Normalize(pixel.B / 255f, Mean[2], Std[2]);
                }
            }
        });

        return chw;
    }

    private static void ApplyAugmentation(Image<Rgb24> image, ImageAugmentationOptions options, Random random)
    {
        image.Mutate(context =>
        {
            var minCropScale = Math.Clamp(options.MinRandomCropScale, 0.5f, 1f);
            if (minCropScale < 0.999f)
            {
                var cropScale = NextFloat(random, minCropScale, 1f);
                var cropWidth = Math.Clamp((int)Math.Round(image.Width * cropScale), 1, image.Width);
                var cropHeight = Math.Clamp((int)Math.Round(image.Height * cropScale), 1, image.Height);

                var maxX = image.Width - cropWidth;
                var maxY = image.Height - cropHeight;
                var cropX = maxX > 0 ? random.Next(maxX + 1) : 0;
                var cropY = maxY > 0 ? random.Next(maxY + 1) : 0;

                context.Crop(new Rectangle(cropX, cropY, cropWidth, cropHeight));
            }

            var flipProbability = Math.Clamp(options.HorizontalFlipProbability, 0f, 1f);
            if (flipProbability > 0f && random.NextDouble() < flipProbability)
            {
                context.Flip(FlipMode.Horizontal);
            }

            var maxRotationDegrees = Math.Max(0f, options.MaxRotationDegrees);
            if (maxRotationDegrees > 0f)
            {
                var angle = NextFloat(random, -maxRotationDegrees, maxRotationDegrees);
                if (Math.Abs(angle) > 0.05f)
                {
                    context.Rotate(angle);
                }
            }

            var brightnessFactor = BuildJitterFactor(random, options.BrightnessJitter);
            if (Math.Abs(brightnessFactor - 1f) > 0.001f)
            {
                context.Brightness(brightnessFactor);
            }

            var contrastFactor = BuildJitterFactor(random, options.ContrastJitter);
            if (Math.Abs(contrastFactor - 1f) > 0.001f)
            {
                context.Contrast(contrastFactor);
            }

            var saturationFactor = BuildJitterFactor(random, options.SaturationJitter);
            if (Math.Abs(saturationFactor - 1f) > 0.001f)
            {
                context.Saturate(saturationFactor);
            }
        });
    }

    private static float BuildJitterFactor(Random random, float jitter)
    {
        var clampedJitter = Math.Clamp(jitter, 0f, 1f);
        if (clampedJitter <= 0f)
        {
            return 1f;
        }

        return NextFloat(random, 1f - clampedJitter, 1f + clampedJitter);
    }

    private static float NextFloat(Random random, float min, float max)
    {
        return (float)(min + (random.NextDouble() * (max - min)));
    }

    private static float Normalize(float value, float mean, float std)
    {
        return (value - mean) / std;
    }

    private static Image<Rgb24> LoadSingleFrameImage(string imagePath)
    {
        var format = Image.DetectFormat(imagePath);
        using var image = Image.Load<Rgb24>(imagePath);

        if (format is GifFormat && image.Frames.Count > 1)
        {
            return image.Frames.CloneFrame(0);
        }

        return image.Clone();
    }
}
