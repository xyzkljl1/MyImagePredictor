using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using ImagePopularity.Core;

namespace ImagePopularity.Trainer;

internal sealed class PreprocessedImageCache : IDisposable
{
    private readonly PreprocessPipelineSpec _pipeline;
    private readonly PreprocessedImageCacheAdapter _adapter;

    public PreprocessedImageCache(string cacheDirectory, PreprocessPipelineSpec pipeline)
    {
        ArgumentNullException.ThrowIfNull(pipeline);
        if (pipeline.ImageSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(pipeline.ImageSize), pipeline.ImageSize, "Image size must be > 0.");
        }

        _pipeline = pipeline;
        _adapter = new PreprocessedImageCacheAdapter(cacheDirectory, pipeline);
    }

    public CacheBuildResult Build(IEnumerable<string> sourceImagePaths, string? progressLabel = null)
    {
        _adapter.EnsureNotDisposedForOwner();
        ArgumentNullException.ThrowIfNull(sourceImagePaths);

        var uniquePaths = sourceImagePaths
            .Where(path => !string.IsNullOrWhiteSpace(path))
            .Select(Path.GetFullPath)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();

        var reused = 0;
        var created = 0;
        var failed = 0;
        using var progress = string.IsNullOrWhiteSpace(progressLabel)
            ? null
            : new ConsoleProgressBar(progressLabel, uniquePaths.Count);

        for (var index = 0; index < uniquePaths.Count; index++)
        {
            var sourcePath = uniquePaths[index];
            try
            {
                var resolution = _adapter.EnsureCachedForBuild(sourcePath);
                if (resolution.WasCreated)
                {
                    created++;
                }
                else
                {
                    reused++;
                }
            }
            catch (Exception ex)
            {
                failed++;
                if (progress is not null)
                {
                    Console.WriteLine();
                }

                Console.WriteLine($"Preprocess failed: {sourcePath} ({ex.GetType().Name}: {ex.Message})");
            }

            progress?.Report(index + 1, $"reused={reused} created={created} failed={failed}");
        }

        progress?.Complete($"reused={reused} created={created} failed={failed}");

        return new CacheBuildResult
        {
            Total = uniquePaths.Count,
            Reused = reused,
            Created = created,
            Failed = failed
        };
    }

    public float[] LoadChw(string sourceImagePath)
    {
        return _adapter.LoadChw(sourceImagePath);
    }

    public void Dispose()
    {
        _adapter.Dispose();
    }

    internal sealed class PreprocessPipelineSpec
    {
        public required int ImageSize { get; init; }

        public int Seed { get; init; }

        public bool UseAugmentation { get; init; }

        public ImageAugmentationOptions? AugmentationOptions { get; init; }

        public string Fingerprint
        {
            get
            {
                var builder = new StringBuilder();
                builder.Append("v1|");
                builder.Append(ImageSize.ToString(CultureInfo.InvariantCulture));
                builder.Append('|');
                builder.Append(Seed.ToString(CultureInfo.InvariantCulture));
                builder.Append('|');
                builder.Append(UseAugmentation ? "1" : "0");

                if (UseAugmentation && AugmentationOptions is not null)
                {
                    builder.Append('|');
                    builder.Append(AugmentationOptions.HorizontalFlipProbability.ToString("R", CultureInfo.InvariantCulture));
                    builder.Append('|');
                    builder.Append(AugmentationOptions.MaxRotationDegrees.ToString("R", CultureInfo.InvariantCulture));
                    builder.Append('|');
                    builder.Append(AugmentationOptions.BrightnessJitter.ToString("R", CultureInfo.InvariantCulture));
                    builder.Append('|');
                    builder.Append(AugmentationOptions.ContrastJitter.ToString("R", CultureInfo.InvariantCulture));
                    builder.Append('|');
                    builder.Append(AugmentationOptions.SaturationJitter.ToString("R", CultureInfo.InvariantCulture));
                    builder.Append('|');
                    builder.Append(AugmentationOptions.MinRandomCropScale.ToString("R", CultureInfo.InvariantCulture));
                }

                return builder.ToString();
            }
        }
    }

    internal sealed class CacheBuildResult
    {
        public int Total { get; init; }

        public int Reused { get; init; }

        public int Created { get; init; }

        public int Failed { get; init; }
    }

    private sealed class PreprocessedImageCacheAdapter : ImagePathTensorCacheBase
    {
        private readonly PreprocessPipelineSpec _pipeline;

        public PreprocessedImageCacheAdapter(string cacheDirectory, PreprocessPipelineSpec pipeline)
            : base(cacheDirectory, pipeline.ImageSize, pipeline.Fingerprint)
        {
            _pipeline = pipeline;
        }

        public void EnsureNotDisposedForOwner()
        {
            EnsureNotDisposed();
        }

        public ShardedTensorCacheStore.CacheResolution EnsureCachedForBuild(string sourceImagePath)
        {
            EnsureNotDisposed();

            var fullPath = Path.GetFullPath(sourceImagePath);
            var resolution = EnsureCached(fullPath);
            RememberResolved(fullPath, resolution.Entry);
            return resolution;
        }

        protected override ShardedTensorCacheStore.CacheResolution EnsureCachedCore(string sourceImagePath, bool forceRebuild)
        {
            var sourceInfo = new FileInfo(sourceImagePath);
            if (!sourceInfo.Exists)
            {
                throw new FileNotFoundException($"Image not found: {sourceImagePath}", sourceImagePath);
            }

            return StoreEnsureCached(
                sourceInfo.FullName,
                new ShardedTensorCacheStore.SourceDescriptor(
                    sourceInfo.FullName,
                    sourceInfo.Length,
                    sourceInfo.LastWriteTimeUtc.Ticks),
                tensorFactory: () =>
                {
                    var random = _pipeline.UseAugmentation
                        ? CreateDeterministicRandom(sourceInfo.FullName, _pipeline.Seed)
                        : null;
                    return ImageTensorFactory.LoadNormalizedChw(
                        sourceInfo.FullName,
                        _pipeline.ImageSize,
                        augmentation: _pipeline.UseAugmentation ? _pipeline.AugmentationOptions : null,
                        random: random);
                },
                forceRebuild: forceRebuild);
        }

        private static Random CreateDeterministicRandom(string sourcePath, int seed)
        {
            var normalizedPath = Path.GetFullPath(sourcePath);
            if (OperatingSystem.IsWindows())
            {
                normalizedPath = normalizedPath.ToUpperInvariant();
            }

            var bytes = Encoding.UTF8.GetBytes($"{seed}|{normalizedPath}");
            var hash = SHA256.HashData(bytes);
            var randomSeed = BitConverter.ToInt32(hash, 0);
            return new Random(randomSeed);
        }
    }
}
