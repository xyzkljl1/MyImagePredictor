namespace ImagePopularity.Core;

internal abstract class ImagePathTensorCacheBase : IDisposable
{
    private readonly Dictionary<string, ResolvedCacheEntry> _resolvedEntries =
        new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, GroupStoreContext> _groupStores =
        new(StringComparer.OrdinalIgnoreCase);
    private readonly string _cacheRoot;
    private bool _disposed;

    protected ImagePathTensorCacheBase(string cacheDirectory, int imageSize, string pipelineFingerprint)
    {
        if (string.IsNullOrWhiteSpace(cacheDirectory))
        {
            throw new ArgumentException("Cache directory cannot be empty.", nameof(cacheDirectory));
        }

        if (imageSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(imageSize), imageSize, "Image size must be > 0.");
        }

        if (string.IsNullOrWhiteSpace(pipelineFingerprint))
        {
            throw new ArgumentException("Pipeline fingerprint cannot be empty.", nameof(pipelineFingerprint));
        }

        _cacheRoot = Path.GetFullPath(cacheDirectory);
        Directory.CreateDirectory(_cacheRoot);

        ImageSize = imageSize;
        PipelineFingerprint = pipelineFingerprint;
    }

    protected int ImageSize { get; }

    protected string PipelineFingerprint { get; }

    protected void EnsureNotDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(GetType().Name);
        }
    }

    protected void RememberResolved(string sourceImagePath, ShardedTensorCacheStore.CacheEntry entry)
    {
        var fullPath = Path.GetFullPath(sourceImagePath);
        var group = ParentDirectoryCacheGroup.FromSourcePath(fullPath);
        _resolvedEntries[fullPath] = new ResolvedCacheEntry(group.LogicalName, group.SafeName, entry);
    }

    protected ShardedTensorCacheStore.CacheResolution EnsureCached(string sourceImagePath, bool forceRebuild = false)
    {
        return EnsureCachedCore(Path.GetFullPath(sourceImagePath), forceRebuild);
    }

    public float[] LoadChw(string sourceImagePath)
    {
        EnsureNotDisposed();

        if (string.IsNullOrWhiteSpace(sourceImagePath))
        {
            throw new ArgumentException("Image path cannot be empty.", nameof(sourceImagePath));
        }

        var fullPath = Path.GetFullPath(sourceImagePath);
        var resolved = Resolve(fullPath);
        var store = GetOrCreateStore(resolved.GroupLogicalName, resolved.GroupSafeName);

        try
        {
            return store.Read(resolved.Entry);
        }
        catch (Exception ex) when (IsRecoverableCacheReadFailure(ex))
        {
            var rebuilt = EnsureCachedCore(fullPath, forceRebuild: true);
            var group = ParentDirectoryCacheGroup.FromSourcePath(fullPath);
            _resolvedEntries[fullPath] = new ResolvedCacheEntry(group.LogicalName, group.SafeName, rebuilt.Entry);
            return GetOrCreateStore(group.LogicalName, group.SafeName).Read(rebuilt.Entry);
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        foreach (var groupStore in _groupStores.Values)
        {
            groupStore.Store.Dispose();
        }

        _groupStores.Clear();
        _disposed = true;
    }

    protected ShardedTensorCacheStore.CacheResolution StoreEnsureCached(
        string sourceImagePath,
        ShardedTensorCacheStore.SourceDescriptor sourceDescriptor,
        Func<float[]> tensorFactory,
        bool forceRebuild)
    {
        var group = ParentDirectoryCacheGroup.FromSourcePath(sourceImagePath);
        return GetOrCreateStore(group.LogicalName, group.SafeName)
            .EnsureCached(sourceDescriptor, tensorFactory, forceRebuild);
    }

    private ResolvedCacheEntry Resolve(string sourceImagePath)
    {
        if (_resolvedEntries.TryGetValue(sourceImagePath, out var resolved))
        {
            return resolved;
        }

        var resolution = EnsureCachedCore(sourceImagePath, forceRebuild: false);
        var group = ParentDirectoryCacheGroup.FromSourcePath(sourceImagePath);
        resolved = new ResolvedCacheEntry(group.LogicalName, group.SafeName, resolution.Entry);
        _resolvedEntries[sourceImagePath] = resolved;
        return resolved;
    }

    private ShardedTensorCacheStore GetOrCreateStore(string groupLogicalName, string groupSafeName)
    {
        if (_groupStores.TryGetValue(groupLogicalName, out var existing))
        {
            return existing.Store;
        }

        var groupCacheDirectory = Path.Combine(_cacheRoot, groupSafeName);
        Directory.CreateDirectory(groupCacheDirectory);

        var store = new ShardedTensorCacheStore(
            groupCacheDirectory,
            ImageSize,
            PipelineFingerprint,
            filePrefix: groupSafeName);
        _groupStores[groupLogicalName] = new GroupStoreContext(store);
        return store;
    }

    protected static bool IsRecoverableCacheReadFailure(Exception ex)
    {
        return ex is IOException or InvalidDataException or UnauthorizedAccessException;
    }

    private readonly record struct ResolvedCacheEntry(
        string GroupLogicalName,
        string GroupSafeName,
        ShardedTensorCacheStore.CacheEntry Entry);

    private sealed class GroupStoreContext
    {
        public GroupStoreContext(ShardedTensorCacheStore store)
        {
            Store = store;
        }

        public ShardedTensorCacheStore Store { get; }
    }

    protected abstract ShardedTensorCacheStore.CacheResolution EnsureCachedCore(string sourceImagePath, bool forceRebuild);
}
