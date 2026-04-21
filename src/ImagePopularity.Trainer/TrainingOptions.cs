using ImagePopularity.Core;
using System.Globalization;

namespace ImagePopularity.Trainer;

internal sealed class TrainingOptions
{
    private static readonly HashSet<string> SupportedOptionKeys = new(StringComparer.OrdinalIgnoreCase)
    {
        "popular-dir",
        "unpopular-dir",
        "output-model",
        "preprocess-cache-dir",
        "train-image-size",
        "epochs",
        "batch-size",
        "learning-rate",
        "fine-tune-learning-rate",
        "weight-decay",
        "validation-split",
        "seed",
        "max-samples-per-class",
        "backbone",
        "pretrained-weights",
        "freeze-backbone-epochs",
        "enable-augmentation",
        "hflip-prob",
        "max-rotation-deg",
        "brightness-jitter",
        "contrast-jitter",
        "saturation-jitter",
        "min-random-crop-scale"
    };

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

    public static TrainingOptions Parse(string[] args)
    {
        if (args.Length == 0 || args.Any(x => x.Equals("--help", StringComparison.OrdinalIgnoreCase)))
        {
            throw new ArgumentException(GetUsage());
        }

        var map = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        for (var i = 0; i < args.Length; i++)
        {
            var token = args[i];
            if (!token.StartsWith("--", StringComparison.Ordinal))
            {
                continue;
            }

            var key = token[2..];
            var value = "true";

            if (i + 1 < args.Length && !args[i + 1].StartsWith("--", StringComparison.Ordinal))
            {
                value = args[i + 1];
                i++;
            }

            map[key] = value;
        }

        if (map.ContainsKey("device"))
        {
            throw new ArgumentException("Option --device is no longer supported. Trainer always uses CUDA.\n\n" + GetUsage());
        }

        if (map.ContainsKey("pretrained"))
        {
            throw new ArgumentException("Option --pretrained is no longer supported. Pretrained backbone is always enabled.\n\n" + GetUsage());
        }

        if (map.ContainsKey("pretrained-cache-dir"))
        {
            throw new ArgumentException("Option --pretrained-cache-dir is no longer supported. Default cache directory is models/pretrained.\n\n" + GetUsage());
        }

        if (map.ContainsKey("image-size"))
        {
            throw new ArgumentException("Option --image-size is no longer supported. Use --train-image-size.\n\n" + GetUsage());
        }

        if (map.ContainsKey("inference-image-size"))
        {
            throw new ArgumentException("Option --inference-image-size is no longer supported. Recommended inference image size now always follows --train-image-size.\n\n" + GetUsage());
        }

        if (map.ContainsKey("enable-preprocess-cache"))
        {
            throw new ArgumentException("Option --enable-preprocess-cache is no longer supported. Trainer always uses preprocess cache.\n\n" + GetUsage());
        }

        var unknownKeys = map.Keys
            .Where(key => !SupportedOptionKeys.Contains(key))
            .OrderBy(key => key, StringComparer.OrdinalIgnoreCase)
            .ToArray();
        if (unknownKeys.Length > 0)
        {
            var formatted = string.Join(", ", unknownKeys.Select(key => $"--{key}"));
            throw new ArgumentException($"Unknown option(s): {formatted}\n\n{GetUsage()}");
        }

        static string Require(Dictionary<string, string> source, string key)
        {
            if (source.TryGetValue(key, out var value) && !string.IsNullOrWhiteSpace(value))
            {
                return value;
            }

            throw new ArgumentException($"Missing required option --{key}\n\n{GetUsage()}");
        }

        var trainImageSize = ReadInt(map, "train-image-size", 320);
        var epochs = ReadInt(map, "epochs", 20);
        var batchSize = ReadInt(map, "batch-size", 128);
        var learningRate = ReadDouble(map, "learning-rate", 3e-4);
        var fineTuneLearningRate = ReadDouble(map, "fine-tune-learning-rate", 5e-5);
        var weightDecay = ReadDouble(map, "weight-decay", 1e-4);
        var validationSplit = ReadDouble(map, "validation-split", 0.1);
        var seed = ReadInt(map, "seed", 42);
        var maxSamplesPerClass = ReadInt(map, "max-samples-per-class", 0);
        var backbone = PopularityModelConfig.NormalizeBackbone(map.TryGetValue("backbone", out var backboneValue) ? backboneValue : PopularityBackboneCatalog.DefaultBackbone);
        var pretrainedWeightsFile = map.TryGetValue("pretrained-weights", out var pretrainedWeights) ? pretrainedWeights : null;
        var freezeBackboneEpochs = Math.Max(0, ReadInt(map, "freeze-backbone-epochs", 3));
        var enableAugmentation = ReadBool(map, "enable-augmentation", true);
        var horizontalFlipProbability = ReadDouble(map, "hflip-prob", 0.5);
        var maxRotationDegrees = ReadDouble(map, "max-rotation-deg", 12);
        var brightnessJitter = ReadDouble(map, "brightness-jitter", 0.15);
        var contrastJitter = ReadDouble(map, "contrast-jitter", 0.15);
        var saturationJitter = ReadDouble(map, "saturation-jitter", 0.15);
        var minRandomCropScale = ReadDouble(map, "min-random-crop-scale", 0.85);
        var outputModelPrefix = ParseOutputModelPrefix(
            map.TryGetValue("output-model", out var outputModel) ? outputModel : null);

        var options = new TrainingOptions
        {
            PopularDirectory = Require(map, "popular-dir"),
            UnpopularDirectory = Require(map, "unpopular-dir"),
            OutputModelPrefix = outputModelPrefix,
            PreprocessCacheDirectory = map.TryGetValue("preprocess-cache-dir", out var preprocessCacheDirectory)
                ? preprocessCacheDirectory
                : Path.Combine("models", "preprocess-cache"),
            TrainImageSize = trainImageSize,
            Epochs = epochs,
            BatchSize = batchSize,
            LearningRate = learningRate,
            FineTuneLearningRate = fineTuneLearningRate,
            WeightDecay = weightDecay,
            ValidationSplit = validationSplit,
            Seed = seed,
            MaxSamplesPerClass = maxSamplesPerClass,
            Backbone = backbone,
            PretrainedWeightsFile = pretrainedWeightsFile,
            FreezeBackboneEpochs = freezeBackboneEpochs,
            EnableAugmentation = enableAugmentation,
            HorizontalFlipProbability = horizontalFlipProbability,
            MaxRotationDegrees = maxRotationDegrees,
            BrightnessJitter = brightnessJitter,
            ContrastJitter = contrastJitter,
            SaturationJitter = saturationJitter,
            MinRandomCropScale = minRandomCropScale
        };

        if (!PopularityModelConfig.IsSupportedBackbone(options.Backbone))
        {
            throw new ArgumentException($"Unsupported backbone: {options.Backbone}. Use {PopularityBackboneCatalog.SupportedList}.");
        }

        if (options.TrainImageSize <= 0)
        {
            throw new ArgumentException("train-image-size must be > 0.");
        }

        if (string.IsNullOrWhiteSpace(options.PreprocessCacheDirectory))
        {
            throw new ArgumentException("preprocess-cache-dir cannot be empty.");
        }

        if (options.Epochs <= 0)
        {
            throw new ArgumentException("epochs must be > 0.");
        }

        if (options.BatchSize <= 0)
        {
            throw new ArgumentException("batch-size must be > 0.");
        }

        if (options.ValidationSplit is <= 0 or >= 1)
        {
            throw new ArgumentException("validation-split must be in (0, 1).");
        }

        if (options.HorizontalFlipProbability is < 0 or > 1)
        {
            throw new ArgumentException("hflip-prob must be in [0, 1].");
        }

        if (options.MinRandomCropScale is <= 0 or > 1)
        {
            throw new ArgumentException("min-random-crop-scale must be in (0, 1].");
        }

        var popularFullPath = Path.GetFullPath(options.PopularDirectory).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        var unpopularFullPath = Path.GetFullPath(options.UnpopularDirectory).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);

        if (string.Equals(popularFullPath, unpopularFullPath, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException("popular-dir and unpopular-dir must be different directories.");
        }

        return options;
    }

    public static string GetUsage()
    {
        return """
Usage:
  dotnet run --project src/ImagePopularity.Trainer -- \
    --popular-dir <path> \
    --unpopular-dir <path> \
    [--output-model <prefix>] \
    [--preprocess-cache-dir models/preprocess-cache] \
    [--train-image-size 320] \
    [--backbone resnet152] \
    [--pretrained-weights <path>] \
    [--freeze-backbone-epochs 3] \
    [--enable-augmentation true] \
    [--hflip-prob 0.5] \
    [--max-rotation-deg 12] \
    [--brightness-jitter 0.15] \
    [--contrast-jitter 0.15] \
    [--saturation-jitter 0.15] \
    [--min-random-crop-scale 0.85] \
    [--epochs 20] \
    [--batch-size 128] \
    [--learning-rate 0.0003] \
    [--fine-tune-learning-rate 0.00005] \
    [--weight-decay 0.0001] \
    [--validation-split 0.1] \
    [--max-samples-per-class 0] \
    [--seed 42]

Notes:
  --popular-dir should contain only popular images (label=1).
  --unpopular-dir should contain only unpopular images (label=0).
  --output-model is only a file-name prefix, not a path. It must not contain
  '\' or '/'. Generated models are always written under models/.
  The trainer always auto-generates the rest of the model file name using train
  sample count, image size, augmentation flag, epochs, batch size, seed, and
  training completion time (month/day/hour/minute).
  Preprocess cache is always enabled for training.
  Pretrained backbone is always enabled.
  If --pretrained-weights is omitted, the trainer will auto-download
  TorchSharp-compatible ImageNet weights.
""";
    }

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
        var timestampSuffix = completedAtLocal is null
            ? string.Empty
            : $"_{completedAtLocal.Value.ToString("MMddHHmm", CultureInfo.InvariantCulture)}";
        return $"{OutputModelPrefix}{trainSampleCount}_{TrainImageSize}_{augmentationTag}_e{Epochs}b{BatchSize}s{Seed}{timestampSuffix}.pt";
    }

    private static string ParseOutputModelPrefix(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return string.Empty;
        }

        var prefix = value.Trim();
        if (string.IsNullOrWhiteSpace(prefix))
        {
            return string.Empty;
        }

        if (prefix.Contains('\\') || prefix.Contains('/'))
        {
            throw new ArgumentException("output-model must be only a file-name prefix and cannot contain '\\' or '/'.");
        }

        if (string.Equals(Path.GetExtension(prefix), ".pt", StringComparison.OrdinalIgnoreCase))
        {
            prefix = Path.GetFileNameWithoutExtension(prefix);
        }

        return prefix;
    }

    private static int ReadInt(Dictionary<string, string> map, string key, int fallback)
    {
        if (!map.TryGetValue(key, out var value))
        {
            return fallback;
        }

        if (int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed))
        {
            return parsed;
        }

        throw new ArgumentException($"Invalid integer for --{key}: {value}\n\n{GetUsage()}");
    }

    private static double ReadDouble(Dictionary<string, string> map, string key, double fallback)
    {
        if (!map.TryGetValue(key, out var value))
        {
            return fallback;
        }

        if (double.TryParse(value, NumberStyles.Float | NumberStyles.AllowThousands, CultureInfo.InvariantCulture, out var parsed))
        {
            return parsed;
        }

        throw new ArgumentException($"Invalid number for --{key}: {value}\n\n{GetUsage()}");
    }

    private static bool ReadBool(Dictionary<string, string> map, string key, bool fallback)
    {
        if (!map.TryGetValue(key, out var value))
        {
            return fallback;
        }

        if (bool.TryParse(value, out var parsed))
        {
            return parsed;
        }

        if (value == "1")
        {
            return true;
        }

        if (value == "0")
        {
            return false;
        }

        throw new ArgumentException($"Invalid boolean for --{key}: {value}. Use true/false or 1/0.\n\n{GetUsage()}");
    }
}
