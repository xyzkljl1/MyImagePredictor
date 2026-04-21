using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace ImagePopularity.Core;

internal sealed class ShardedTensorCacheStore : IDisposable
{
    private const ulong IndexMagic = 0x58444E4945484341UL;
    private const ulong RecordMagic = 0x3144524F43455254UL;
    private const ulong RecordFooterMagic = 0x31444E45434E4554UL;
    private const int IndexFormatVersion = 1;
    private const int RecordFormatVersion = 1;
    private const int KeyTextLength = 64;
    private const int HashLength = 32;
    private const int MaxIndexPayloadBytes = 64 * 1024;
    private const long DefaultTargetShardSizeBytes = 1L * 1024 * 1024 * 1024;
    private const int ReadBufferSize = 1024 * 1024;
    private const int RecordHeaderSize =
        sizeof(ulong) +
        sizeof(int) +
        KeyTextLength +
        sizeof(int) +
        sizeof(int) +
        sizeof(int) +
        HashLength;

    private readonly string _cacheRoot;
    private readonly int _imageSize;
    private readonly string _pipelineFingerprint;
    private readonly long _targetShardSizeBytes;
    private readonly string _filePrefix;
    private readonly string _shardSearchPattern;
    private readonly string _indexPath;
    private readonly Dictionary<string, CacheEntry> _entries = new(StringComparer.Ordinal);
    private readonly Dictionary<int, string> _shardPaths = new();
    private readonly Dictionary<int, FileStream> _readStreams = new();
    private readonly FileStream _indexStream;
    private readonly BinaryWriter _indexWriter;

    private FileStream _activeShardWriteStream = null!;
    private BinaryWriter _activeShardWriter = null!;
    private int _activeShardId;
    private bool _disposed;

    public ShardedTensorCacheStore(
        string cacheDirectory,
        int imageSize,
        string pipelineFingerprint,
        long targetShardSizeBytes = DefaultTargetShardSizeBytes,
        string? filePrefix = null)
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

        if (targetShardSizeBytes <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(targetShardSizeBytes), targetShardSizeBytes, "Target shard size must be > 0.");
        }

        _cacheRoot = Path.GetFullPath(cacheDirectory);
        _imageSize = imageSize;
        _pipelineFingerprint = pipelineFingerprint;
        _targetShardSizeBytes = targetShardSizeBytes;
        _filePrefix = NormalizeFilePrefix(filePrefix);
        _shardSearchPattern = string.IsNullOrEmpty(_filePrefix)
            ? "shard-*.bin"
            : $"{_filePrefix}-shard-*.bin";
        _indexPath = Path.Combine(
            _cacheRoot,
            string.IsNullOrEmpty(_filePrefix) ? "index.log" : $"{_filePrefix}-index.log");

        Directory.CreateDirectory(_cacheRoot);

        _indexStream = new FileStream(
            _indexPath,
            FileMode.OpenOrCreate,
            FileAccess.ReadWrite,
            FileShare.Read,
            bufferSize: 64 * 1024,
            FileOptions.None);

        LoadIndexAndRepair();

        _indexStream.Seek(0, SeekOrigin.End);
        _indexWriter = new BinaryWriter(_indexStream, Encoding.UTF8, leaveOpen: true);

        OpenActiveShardForAppend(_activeShardId);
    }

    public string CacheRoot => _cacheRoot;

    public CacheResolution EnsureCached(SourceDescriptor sourceDescriptor, Func<float[]> tensorFactory, bool forceRebuild = false)
    {
        EnsureNotDisposed();
        ArgumentNullException.ThrowIfNull(tensorFactory);

        var normalizedSource = NormalizeSource(sourceDescriptor);
        var key = ComputeKey(normalizedSource.SourcePath);

        if (!forceRebuild &&
            _entries.TryGetValue(key, out var existing) &&
            existing.Matches(normalizedSource, _pipelineFingerprint))
        {
            return new CacheResolution(existing, WasCreated: false);
        }

        var chw = tensorFactory();
        ArgumentNullException.ThrowIfNull(chw);

        var expectedLength = 3 * _imageSize * _imageSize;
        if (chw.Length != expectedLength)
        {
            throw new InvalidDataException($"Tensor length mismatch: expected {expectedLength}, got {chw.Length}.");
        }

        var entry = AppendRecord(normalizedSource, key, chw);
        _entries[key] = entry;
        return new CacheResolution(entry, WasCreated: true);
    }

    public float[] Read(CacheEntry entry)
    {
        EnsureNotDisposed();

        if (!string.Equals(entry.PipelineFingerprint, _pipelineFingerprint, StringComparison.Ordinal))
        {
            throw new InvalidDataException($"Pipeline fingerprint mismatch for key {entry.Key}.");
        }

        if (entry.RecordLength <= 0)
        {
            throw new InvalidDataException($"Record length must be > 0 for key {entry.Key}.");
        }

        var stream = GetReadStream(entry.ShardId);
        lock (stream)
        {
            return ReadRecord(stream, entry);
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        foreach (var stream in _readStreams.Values)
        {
            stream.Dispose();
        }

        _readStreams.Clear();

        _activeShardWriter.Dispose();
        _activeShardWriteStream.Dispose();
        _indexWriter.Dispose();
        _indexStream.Dispose();

        _disposed = true;
    }

    private void LoadIndexAndRepair()
    {
        var maxExistingShardId = -1;
        var validIndexLength = 0L;
        var maxPublishedEnds = new Dictionary<int, long>();

        foreach (var shardPath in Directory.EnumerateFiles(_cacheRoot, _shardSearchPattern, SearchOption.TopDirectoryOnly))
        {
            if (TryParseShardId(shardPath, out var shardId))
            {
                _shardPaths[shardId] = shardPath;
                maxExistingShardId = Math.Max(maxExistingShardId, shardId);
            }
        }

        _indexStream.Seek(0, SeekOrigin.Begin);
        using (var reader = new BinaryReader(_indexStream, Encoding.UTF8, leaveOpen: true))
        {
            while (_indexStream.Position < _indexStream.Length)
            {
                var entryStart = _indexStream.Position;
                if (!TryReadIndexEntry(reader, out var payload))
                {
                    break;
                }

                validIndexLength = _indexStream.Position;

                var shardEnd = checked(payload.Offset + payload.RecordLength);
                if (!maxPublishedEnds.TryGetValue(payload.ShardId, out var currentEnd) || shardEnd > currentEnd)
                {
                    maxPublishedEnds[payload.ShardId] = shardEnd;
                }

                maxExistingShardId = Math.Max(maxExistingShardId, payload.ShardId);
                _shardPaths[payload.ShardId] = GetShardPath(payload.ShardId);

                if (string.Equals(payload.PipelineFingerprint, _pipelineFingerprint, StringComparison.Ordinal))
                {
                    _entries[payload.Key] = new CacheEntry(
                        payload.Key,
                        payload.SourcePath,
                        payload.SourceLength,
                        payload.SourceLastWriteUtcTicks,
                        payload.PipelineFingerprint,
                        payload.ShardId,
                        payload.Offset,
                        payload.RecordLength);
                }

                validIndexLength = _indexStream.Position;
            }
        }

        if (_indexStream.Length != validIndexLength)
        {
            _indexStream.SetLength(validIndexLength);
        }

        foreach (var pair in _shardPaths.ToArray())
        {
            var shardPath = pair.Value;
            if (!File.Exists(shardPath))
            {
                continue;
            }

            var publishedEnd = maxPublishedEnds.TryGetValue(pair.Key, out var value)
                ? value
                : 0L;

            using var stream = new FileStream(
                shardPath,
                FileMode.Open,
                FileAccess.ReadWrite,
                FileShare.Read,
                bufferSize: ReadBufferSize,
                FileOptions.None);

            if (stream.Length > publishedEnd)
            {
                stream.SetLength(publishedEnd);
            }
        }

        _activeShardId = Math.Max(maxExistingShardId, 0);
    }

    private CacheEntry AppendRecord(SourceDescriptor sourceDescriptor, string key, float[] chw)
    {
        var payloadBytes = new byte[chw.Length * sizeof(float)];
        Buffer.BlockCopy(chw, 0, payloadBytes, 0, payloadBytes.Length);
        var payloadHash = SHA256.HashData(payloadBytes);
        var keyBytes = Encoding.ASCII.GetBytes(key);

        if (keyBytes.Length != KeyTextLength)
        {
            throw new InvalidDataException($"Cache key must be {KeyTextLength} ASCII characters.");
        }

        var recordLength = checked(RecordHeaderSize + payloadBytes.Length + sizeof(ulong));
        RotateActiveShardIfNeeded(recordLength);

        _activeShardWriteStream.Seek(0, SeekOrigin.End);
        var offset = _activeShardWriteStream.Position;

        _activeShardWriter.Write(RecordMagic);
        _activeShardWriter.Write(RecordFormatVersion);
        _activeShardWriter.Write(keyBytes);
        _activeShardWriter.Write(_imageSize);
        _activeShardWriter.Write(chw.Length);
        _activeShardWriter.Write(payloadBytes.Length);
        _activeShardWriter.Write(payloadHash);
        _activeShardWriter.Write(payloadBytes);
        _activeShardWriter.Write(RecordFooterMagic);
        _activeShardWriter.Flush();
        _activeShardWriteStream.Flush(flushToDisk: true);

        var entry = new CacheEntry(
            key,
            sourceDescriptor.SourcePath,
            sourceDescriptor.SourceLength,
            sourceDescriptor.SourceLastWriteUtcTicks,
            _pipelineFingerprint,
            _activeShardId,
            offset,
            recordLength);

        AppendIndexEntry(entry);
        return entry;
    }

    private void AppendIndexEntry(CacheEntry entry)
    {
        var payload = new IndexEntryPayload
        {
            Key = entry.Key,
            SourcePath = entry.SourcePath,
            SourceLength = entry.SourceLength,
            SourceLastWriteUtcTicks = entry.SourceLastWriteUtcTicks,
            PipelineFingerprint = entry.PipelineFingerprint,
            ShardId = entry.ShardId,
            Offset = entry.Offset,
            RecordLength = entry.RecordLength
        };

        var payloadBytes = JsonSerializer.SerializeToUtf8Bytes(payload);
        if (payloadBytes.Length <= 0 || payloadBytes.Length > MaxIndexPayloadBytes)
        {
            throw new InvalidDataException($"Index payload length is invalid: {payloadBytes.Length}.");
        }

        var payloadHash = SHA256.HashData(payloadBytes);

        _indexWriter.Write(IndexMagic);
        _indexWriter.Write(IndexFormatVersion);
        _indexWriter.Write(payloadBytes.Length);
        _indexWriter.Write(payloadBytes);
        _indexWriter.Write(payloadHash);
        _indexWriter.Flush();
        _indexStream.Flush(flushToDisk: true);
    }

    private bool TryReadIndexEntry(BinaryReader reader, out IndexEntryPayload payload)
    {
        payload = default!;

        try
        {
            var magic = reader.ReadUInt64();
            if (magic != IndexMagic)
            {
                return false;
            }

            var version = reader.ReadInt32();
            if (version != IndexFormatVersion)
            {
                return false;
            }

            var payloadLength = reader.ReadInt32();
            if (payloadLength <= 0 || payloadLength > MaxIndexPayloadBytes)
            {
                return false;
            }

            var payloadBytes = reader.ReadBytes(payloadLength);
            if (payloadBytes.Length != payloadLength)
            {
                return false;
            }

            var storedHash = reader.ReadBytes(HashLength);
            if (storedHash.Length != HashLength)
            {
                return false;
            }

            var actualHash = SHA256.HashData(payloadBytes);
            if (!storedHash.AsSpan().SequenceEqual(actualHash))
            {
                return false;
            }

            var parsed = JsonSerializer.Deserialize<IndexEntryPayload>(payloadBytes);
            if (parsed is null ||
                string.IsNullOrWhiteSpace(parsed.Key) ||
                parsed.Key.Length != KeyTextLength ||
                string.IsNullOrWhiteSpace(parsed.SourcePath) ||
                string.IsNullOrWhiteSpace(parsed.PipelineFingerprint) ||
                parsed.ShardId < 0 ||
                parsed.Offset < 0 ||
                parsed.RecordLength <= 0)
            {
                return false;
            }

            payload = parsed;
            return true;
        }
        catch (EndOfStreamException)
        {
            return false;
        }
    }

    private float[] ReadRecord(FileStream stream, CacheEntry entry)
    {
        stream.Seek(entry.Offset, SeekOrigin.Begin);

        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);

        var magic = reader.ReadUInt64();
        if (magic != RecordMagic)
        {
            throw new InvalidDataException($"Record magic mismatch for key {entry.Key}.");
        }

        var version = reader.ReadInt32();
        if (version != RecordFormatVersion)
        {
            throw new InvalidDataException($"Record version mismatch for key {entry.Key}: {version}.");
        }

        var keyBytes = reader.ReadBytes(KeyTextLength);
        if (keyBytes.Length != KeyTextLength)
        {
            throw new EndOfStreamException($"Record key is truncated for key {entry.Key}.");
        }

        var actualKey = Encoding.ASCII.GetString(keyBytes);
        if (!string.Equals(actualKey, entry.Key, StringComparison.Ordinal))
        {
            throw new InvalidDataException($"Record key mismatch for cache entry {entry.Key}.");
        }

        var imageSize = reader.ReadInt32();
        if (imageSize != _imageSize)
        {
            throw new InvalidDataException($"Record image size mismatch: expected {_imageSize}, got {imageSize}.");
        }

        var floatCount = reader.ReadInt32();
        var expectedFloatCount = 3 * _imageSize * _imageSize;
        if (floatCount != expectedFloatCount)
        {
            throw new InvalidDataException($"Record float count mismatch: expected {expectedFloatCount}, got {floatCount}.");
        }

        var payloadLength = reader.ReadInt32();
        var expectedPayloadLength = checked(floatCount * sizeof(float));
        if (payloadLength != expectedPayloadLength)
        {
            throw new InvalidDataException($"Record payload length mismatch: expected {expectedPayloadLength}, got {payloadLength}.");
        }

        var expectedRecordLength = checked(RecordHeaderSize + payloadLength + sizeof(ulong));
        if (entry.RecordLength != expectedRecordLength)
        {
            throw new InvalidDataException($"Record length mismatch for key {entry.Key}: expected {entry.RecordLength}, actual {expectedRecordLength}.");
        }

        var storedHash = reader.ReadBytes(HashLength);
        if (storedHash.Length != HashLength)
        {
            throw new EndOfStreamException($"Record hash is truncated for key {entry.Key}.");
        }

        var payloadBytes = reader.ReadBytes(payloadLength);
        if (payloadBytes.Length != payloadLength)
        {
            throw new EndOfStreamException($"Record payload is truncated for key {entry.Key}.");
        }

        var actualHash = SHA256.HashData(payloadBytes);
        if (!storedHash.AsSpan().SequenceEqual(actualHash))
        {
            throw new InvalidDataException($"Record payload hash mismatch for key {entry.Key}.");
        }

        var footer = reader.ReadUInt64();
        if (footer != RecordFooterMagic)
        {
            throw new InvalidDataException($"Record footer mismatch for key {entry.Key}.");
        }

        var chw = new float[floatCount];
        Buffer.BlockCopy(payloadBytes, 0, chw, 0, payloadLength);
        return chw;
    }

    private void RotateActiveShardIfNeeded(int recordLength)
    {
        if (_activeShardWriteStream.Length == 0)
        {
            return;
        }

        if (_activeShardWriteStream.Length + recordLength <= _targetShardSizeBytes)
        {
            return;
        }

        OpenActiveShardForAppend(_activeShardId + 1);
    }

    private void OpenActiveShardForAppend(int shardId)
    {
        _activeShardWriter?.Dispose();
        _activeShardWriteStream?.Dispose();

        var shardPath = GetShardPath(shardId);
        _shardPaths[shardId] = shardPath;

        _activeShardWriteStream = new FileStream(
            shardPath,
            FileMode.OpenOrCreate,
            FileAccess.ReadWrite,
            FileShare.Read,
            bufferSize: ReadBufferSize,
            FileOptions.None);
        _activeShardWriteStream.Seek(0, SeekOrigin.End);
        _activeShardWriter = new BinaryWriter(_activeShardWriteStream, Encoding.UTF8, leaveOpen: true);
        _activeShardId = shardId;
    }

    private FileStream GetReadStream(int shardId)
    {
        if (_readStreams.TryGetValue(shardId, out var existing))
        {
            return existing;
        }

        var shardPath = GetShardPath(shardId);
        if (!File.Exists(shardPath))
        {
            throw new FileNotFoundException($"Shard file not found: {shardPath}", shardPath);
        }

        var stream = new FileStream(
            shardPath,
            FileMode.Open,
            FileAccess.Read,
            FileShare.ReadWrite,
            bufferSize: ReadBufferSize,
            FileOptions.SequentialScan);
        _readStreams[shardId] = stream;
        return stream;
    }

    private string GetShardPath(int shardId)
    {
        if (_shardPaths.TryGetValue(shardId, out var existing))
        {
            return existing;
        }

        var fileName = string.IsNullOrEmpty(_filePrefix)
            ? $"shard-{shardId:D6}.bin"
            : $"{_filePrefix}-shard-{shardId:D6}.bin";
        var path = Path.Combine(_cacheRoot, fileName);
        _shardPaths[shardId] = path;
        return path;
    }

    private bool TryParseShardId(string shardPath, out int shardId)
    {
        shardId = 0;

        var fileName = Path.GetFileNameWithoutExtension(shardPath);
        var prefix = string.IsNullOrEmpty(_filePrefix)
            ? "shard-"
            : $"{_filePrefix}-shard-";
        if (!fileName.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        return int.TryParse(fileName[prefix.Length..], out shardId);
    }

    private static string NormalizeFilePrefix(string? filePrefix)
    {
        if (string.IsNullOrWhiteSpace(filePrefix))
        {
            return string.Empty;
        }

        var builder = new StringBuilder(filePrefix.Length);
        var invalidChars = Path.GetInvalidFileNameChars();

        foreach (var ch in filePrefix)
        {
            if (char.IsControl(ch) || invalidChars.Contains(ch))
            {
                builder.Append('_');
            }
            else
            {
                builder.Append(ch);
            }
        }

        var normalized = builder
            .ToString()
            .Trim()
            .TrimEnd('.');

        return string.IsNullOrWhiteSpace(normalized) ? "cache" : normalized;
    }

    private string ComputeKey(string sourcePath)
    {
        var normalizedPath = Path.GetFullPath(sourcePath);
        if (OperatingSystem.IsWindows())
        {
            normalizedPath = normalizedPath.ToUpperInvariant();
        }

        var bytes = Encoding.UTF8.GetBytes($"{normalizedPath}|{_pipelineFingerprint}");
        var hash = SHA256.HashData(bytes);
        return Convert.ToHexString(hash).ToLowerInvariant();
    }

    private static SourceDescriptor NormalizeSource(SourceDescriptor sourceDescriptor)
    {
        if (string.IsNullOrWhiteSpace(sourceDescriptor.SourcePath))
        {
            throw new ArgumentException("Source path cannot be empty.", nameof(sourceDescriptor));
        }

        return new SourceDescriptor(
            Path.GetFullPath(sourceDescriptor.SourcePath),
            sourceDescriptor.SourceLength,
            sourceDescriptor.SourceLastWriteUtcTicks);
    }

    private void EnsureNotDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(ShardedTensorCacheStore));
        }
    }

    internal readonly record struct SourceDescriptor(string SourcePath, long SourceLength, long SourceLastWriteUtcTicks);

    internal readonly record struct CacheResolution(CacheEntry Entry, bool WasCreated);

    internal readonly record struct CacheEntry(
        string Key,
        string SourcePath,
        long SourceLength,
        long SourceLastWriteUtcTicks,
        string PipelineFingerprint,
        int ShardId,
        long Offset,
        int RecordLength)
    {
        public bool Matches(SourceDescriptor sourceDescriptor, string pipelineFingerprint)
        {
            return string.Equals(SourcePath, sourceDescriptor.SourcePath, StringComparison.OrdinalIgnoreCase) &&
                   SourceLength == sourceDescriptor.SourceLength &&
                   SourceLastWriteUtcTicks == sourceDescriptor.SourceLastWriteUtcTicks &&
                   string.Equals(PipelineFingerprint, pipelineFingerprint, StringComparison.Ordinal);
        }
    }

    private sealed class IndexEntryPayload
    {
        public required string Key { get; init; }

        public required string SourcePath { get; init; }

        public long SourceLength { get; init; }

        public long SourceLastWriteUtcTicks { get; init; }

        public required string PipelineFingerprint { get; init; }

        public int ShardId { get; init; }

        public long Offset { get; init; }

        public int RecordLength { get; init; }
    }
}
