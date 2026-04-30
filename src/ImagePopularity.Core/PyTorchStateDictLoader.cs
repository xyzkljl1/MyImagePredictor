using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using TorchSharp;
using static TorchSharp.torch;

namespace ImagePopularity.Core;

internal static class PyTorchStateDictLoader
{
    private const int FormatVersion = 2;
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        WriteIndented = true
    };

    public static bool TryLoadFloat32StateDict(nn.Module<Tensor, Tensor> module, string weightsFile, string modelDisplayName)
    {
        if (!weightsFile.EndsWith(".pth", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        try
        {
            var sourcePath = Path.GetFullPath(weightsFile);
            var cacheDirectory = sourcePath + $".torchsharp.v{FormatVersion}";
            var cachePaths = EnsureConvertedCache(sourcePath, cacheDirectory, modelDisplayName);
            LoadConvertedStateDict(module, sourcePath, cachePaths.ManifestPath, cachePaths.DataPath, modelDisplayName);
            return true;
        }
        catch (Exception ex)
        {
            Console.WriteLine(
                $"Warning: failed to load pretrained {modelDisplayName} weights from official PyTorch state_dict. Continuing with random {modelDisplayName} initialization. File: {Path.GetFullPath(weightsFile)}. Details: {ex.Message}");
            return false;
        }
    }

    private static ConvertedCachePaths EnsureConvertedCache(
        string sourcePath,
        string cacheDirectory,
        string modelDisplayName)
    {
        var manifestPath = Path.Combine(cacheDirectory, "manifest.json");
        var dataPath = Path.Combine(cacheDirectory, "data.bin");
        if (TryReadManifest(manifestPath, out var existingManifest) &&
            existingManifest is not null &&
            File.Exists(dataPath) &&
            ManifestMatchesSource(existingManifest, sourcePath))
        {
            Console.WriteLine($"Using cached converted {modelDisplayName} weights: {cacheDirectory}");
            return new ConvertedCachePaths(manifestPath, dataPath);
        }

        var scriptPath = ResolveConverterScriptPath();
        var tempDirectory = cacheDirectory + ".tmp." + Guid.NewGuid().ToString("N");
        var keepTemporaryCache = false;

        Directory.CreateDirectory(tempDirectory);

        try
        {
            RunConverterScript(scriptPath, sourcePath, tempDirectory);

            try
            {
                if (Directory.Exists(cacheDirectory))
                {
                    Directory.Delete(cacheDirectory, recursive: true);
                }

                Directory.Move(tempDirectory, cacheDirectory);
                Console.WriteLine($"Converted official PyTorch {modelDisplayName} weights: {cacheDirectory}");
                return new ConvertedCachePaths(
                    Path.Combine(cacheDirectory, "manifest.json"),
                    Path.Combine(cacheDirectory, "data.bin"));
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                keepTemporaryCache = true;
                Console.WriteLine(
                    $"Warning: could not promote converted {modelDisplayName} cache into shared directory. Using temporary converted cache for this run. Cache: {cacheDirectory}. Details: {ex.Message}");
                return new ConvertedCachePaths(
                    Path.Combine(tempDirectory, "manifest.json"),
                    Path.Combine(tempDirectory, "data.bin"));
            }
        }
        finally
        {
            if (!keepTemporaryCache && Directory.Exists(tempDirectory))
            {
                Directory.Delete(tempDirectory, recursive: true);
            }
        }
    }

    private static void LoadConvertedStateDict(
        nn.Module<Tensor, Tensor> module,
        string sourcePath,
        string manifestPath,
        string dataPath,
        string modelDisplayName)
    {
        if (!TryReadManifest(manifestPath, out var manifest) || manifest is null)
        {
            throw new InvalidOperationException($"Converted manifest not found or invalid: {manifestPath}");
        }

        if (!ManifestMatchesSource(manifest, sourcePath))
        {
            throw new InvalidOperationException($"Converted manifest does not match source weights file: {sourcePath}");
        }

        if (manifest.FormatVersion != FormatVersion)
        {
            throw new InvalidOperationException(
                $"Unsupported converted manifest format version {manifest.FormatVersion}. Expected {FormatVersion}.");
        }

        if (!File.Exists(dataPath))
        {
            throw new FileNotFoundException($"Converted data file not found: {dataPath}", dataPath);
        }

        var stateDict = module.state_dict();
        var missingKeys = new List<string>();
        var mismatchedKeys = new List<string>();
        var loadedCount = 0;

        using var noGradScope = no_grad();
        using var dataStream = OpenConvertedDataStream(dataPath);

        foreach (var entry in manifest.Tensors)
        {
            if (!string.Equals(entry.DType, "float32", StringComparison.Ordinal))
            {
                throw new NotSupportedException(
                    $"Unsupported tensor dtype '{entry.DType}' for key '{entry.Name}'. Only float32 is supported.");
            }

            if (!stateDict.ContainsKey(entry.Name))
            {
                missingKeys.Add(entry.Name);
                continue;
            }

            var targetTensor = stateDict[entry.Name];
            var targetShape = targetTensor.shape;
            var expectedShape = entry.Shape.Select(static x => (long)x).ToArray();

            if (!targetShape.SequenceEqual(expectedShape))
            {
                mismatchedKeys.Add(
                    $"{entry.Name} (model={FormatShape(targetShape)}, file={FormatShape(expectedShape)})");
                continue;
            }

            var values = ReadFloat32TensorValues(dataStream, entry);
            using var sourceTensor = tensor(values, dtype: ScalarType.Float32).reshape(expectedShape);
            targetTensor.copy_(sourceTensor);
            loadedCount++;
        }

        if (missingKeys.Count > 0 || mismatchedKeys.Count > 0)
        {
            var details = new List<string>();
            if (missingKeys.Count > 0)
            {
                details.Add($"missing keys: {string.Join(", ", missingKeys.Take(5))}{(missingKeys.Count > 5 ? "..." : string.Empty)}");
            }

            if (mismatchedKeys.Count > 0)
            {
                details.Add($"shape mismatches: {string.Join(", ", mismatchedKeys.Take(5))}{(mismatchedKeys.Count > 5 ? "..." : string.Empty)}");
            }

            throw new InvalidOperationException(
                $"Converted state_dict could not be fully loaded ({string.Join("; ", details)}).");
        }

        Console.WriteLine(
            $"Loaded pretrained {modelDisplayName} weights: {sourcePath} ({loadedCount.ToString(CultureInfo.InvariantCulture)} tensors)");
    }

    private static float[] ReadFloat32TensorValues(FileStream dataStream, ManifestTensorEntry entry)
    {
        checked
        {
            var byteCount = entry.ByteCount;
            var buffer = new byte[byteCount];
            dataStream.Seek(entry.DataOffset, SeekOrigin.Begin);

            var totalRead = 0;
            while (totalRead < byteCount)
            {
                var read = dataStream.Read(buffer, totalRead, byteCount - totalRead);
                if (read <= 0)
                {
                    throw new EndOfStreamException(
                        $"Unexpected end of converted tensor data for '{entry.Name}'. Expected {byteCount} bytes, got {totalRead}.");
                }

                totalRead += read;
            }

            var values = new float[entry.ElementCount];
            Buffer.BlockCopy(buffer, 0, values, 0, byteCount);
            return values;
        }
    }

    private static bool TryReadManifest(string manifestPath, out ConvertedManifest? manifest)
    {
        manifest = null;
        if (!File.Exists(manifestPath))
        {
            return false;
        }

        try
        {
            manifest = JsonSerializer.Deserialize<ConvertedManifest>(File.ReadAllText(manifestPath), JsonOptions);
            return manifest is not null;
        }
        catch
        {
            return false;
        }
    }

    private static bool ManifestMatchesSource(ConvertedManifest manifest, string sourcePath)
    {
        var sourceInfo = new FileInfo(sourcePath);
        return string.Equals(Path.GetFullPath(manifest.SourcePath), sourceInfo.FullName, StringComparison.OrdinalIgnoreCase) &&
               manifest.SourceLength == sourceInfo.Length &&
               manifest.SourceLastWriteTimeUtcTicks == sourceInfo.LastWriteTimeUtc.Ticks;
    }

    private static string ResolveConverterScriptPath()
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "Tools", "convert_pytorch_state_dict.py"),
            Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "src", "ImagePopularity.Core", "Tools", "convert_pytorch_state_dict.py"))
        };

        foreach (var candidate in candidates)
        {
            if (File.Exists(candidate))
            {
                return candidate;
            }
        }

        throw new FileNotFoundException("PyTorch state_dict converter script not found.");
    }

    private static void RunConverterScript(string scriptPath, string sourcePath, string outputDirectory)
    {
        using var process = new Process();
        process.StartInfo = new ProcessStartInfo
        {
            FileName = "python",
            RedirectStandardError = true,
            RedirectStandardOutput = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };
        process.StartInfo.ArgumentList.Add(scriptPath);
        process.StartInfo.ArgumentList.Add(sourcePath);
        process.StartInfo.ArgumentList.Add(outputDirectory);

        process.Start();
        var stdout = process.StandardOutput.ReadToEnd();
        var stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();

        if (!string.IsNullOrWhiteSpace(stdout))
        {
            foreach (var line in stdout.Split(Environment.NewLine, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            {
                Console.WriteLine(line);
            }
        }

        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException(
                $"Converter script failed with exit code {process.ExitCode}. {stderr}".Trim());
        }
    }

    private static string FormatShape(IReadOnlyList<long> shape)
    {
        return string.Join("x", shape);
    }

    private sealed record ConvertedCachePaths(string ManifestPath, string DataPath);

    private static FileStream OpenConvertedDataStream(string dataPath)
    {
        const int maxAttempts = 10;
        var delay = TimeSpan.FromMilliseconds(200);
        Exception? lastException = null;

        for (var attempt = 1; attempt <= maxAttempts; attempt++)
        {
            try
            {
                return new FileStream(
                    dataPath,
                    FileMode.Open,
                    FileAccess.Read,
                    FileShare.ReadWrite | FileShare.Delete);
            }
            catch (UnauthorizedAccessException ex)
            {
                lastException = ex;
            }
            catch (IOException ex)
            {
                lastException = ex;
            }

            Thread.Sleep(delay);
        }

        throw new IOException($"Unable to open converted tensor data file after {maxAttempts} attempts: {dataPath}", lastException);
    }

    private sealed class ConvertedManifest
    {
        [JsonPropertyName("formatVersion")]
        public int FormatVersion { get; init; }

        [JsonPropertyName("sourcePath")]
        public required string SourcePath { get; init; }

        [JsonPropertyName("sourceLength")]
        public long SourceLength { get; init; }

        [JsonPropertyName("sourceLastWriteTimeUtcTicks")]
        public long SourceLastWriteTimeUtcTicks { get; init; }

        [JsonPropertyName("tensors")]
        public required List<ManifestTensorEntry> Tensors { get; init; }
    }

    private sealed class ManifestTensorEntry
    {
        [JsonPropertyName("name")]
        public required string Name { get; init; }

        [JsonPropertyName("dtype")]
        public required string DType { get; init; }

        [JsonPropertyName("shape")]
        public required int[] Shape { get; init; }

        [JsonPropertyName("elementCount")]
        public int ElementCount { get; init; }

        [JsonPropertyName("dataOffset")]
        public long DataOffset { get; init; }

        [JsonPropertyName("byteCount")]
        public int ByteCount { get; init; }
    }
}
