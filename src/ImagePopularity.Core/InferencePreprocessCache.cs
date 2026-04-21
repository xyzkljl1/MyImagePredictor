namespace ImagePopularity.Core;

internal sealed class InferencePreprocessCache : ImagePathTensorCacheBase
{
    public InferencePreprocessCache(string cacheDirectory, int imageSize)
        : base(
            cacheDirectory,
            imageSize,
            $"v1|inference|size={imageSize.ToString(System.Globalization.CultureInfo.InvariantCulture)}")
    {
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
            tensorFactory: () => ImageTensorFactory.LoadNormalizedChw(sourceInfo.FullName, ImageSize),
            forceRebuild: forceRebuild);
    }
}
