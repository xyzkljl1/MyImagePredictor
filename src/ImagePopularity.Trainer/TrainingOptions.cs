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
        "validation-dir",
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
        "min-random-crop-scale",
        "enable-group-aware-training"
    };

    public required IReadOnlyList<string> PopularDirectories { get; init; }

    public required IReadOnlyList<string> UnpopularDirectories { get; init; }

    public string OutputModelPrefix { get; init; } = string.Empty;

    public string PreprocessCacheDirectory { get; init; } = Path.Combine("models", "preprocess-cache");

    public int TrainImageSize { get; init; } = 320;

    public int Epochs { get; init; } = 20;

    public int BatchSize { get; init; } = 128;

    public double LearningRate { get; init; } = 3e-4;

    public double FineTuneLearningRate { get; init; } = 5e-5;

    public double WeightDecay { get; init; } = 1e-4;

    public IReadOnlyList<string> ValidationFileNames { get; init; } = Array.Empty<string>();

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

    public bool EnableGroupAwareTraining { get; init; } = true;

    public static TrainingOptions Parse(string[] args)
    {
        if (args.Length == 0 || args.Any(x => x.Equals("--help", StringComparison.OrdinalIgnoreCase)))
        {
            throw new ArgumentException(GetUsage());
        }

        var map = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        var popularDirectoryValues = new List<string>();
        var unpopularDirectoryValues = new List<string>();
        var validationFileNameValues = new List<string>();

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

            if (string.Equals(key, "popular-dir", StringComparison.OrdinalIgnoreCase))
            {
                popularDirectoryValues.Add(value);
                continue;
            }

            if (string.Equals(key, "unpopular-dir", StringComparison.OrdinalIgnoreCase))
            {
                unpopularDirectoryValues.Add(value);
                continue;
            }

            if (string.Equals(key, "validation-dir", StringComparison.OrdinalIgnoreCase))
            {
                validationFileNameValues.Add(value);
                continue;
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

        var trainImageSize = ReadInt(map, "train-image-size", 320);
        var epochs = ReadInt(map, "epochs", 20);
        var batchSize = ReadInt(map, "batch-size", 128);
        var learningRate = ReadDouble(map, "learning-rate", 3e-4);
        var fineTuneLearningRate = ReadDouble(map, "fine-tune-learning-rate", 5e-5);
        var weightDecay = ReadDouble(map, "weight-decay", 1e-4);
        var popularDirectories = ParseDirectoryList(popularDirectoryValues, "popular-dir");
        var unpopularDirectories = ParseDirectoryList(unpopularDirectoryValues, "unpopular-dir");
        var validationFileNames = ParseValidationFileNames(validationFileNameValues);
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
        var enableGroupAwareTraining = ReadBool(map, "enable-group-aware-training", true);
        var outputModelPrefix = ParseOutputModelPrefix(
            map.TryGetValue("output-model", out var outputModel) ? outputModel : null);

        var options = new TrainingOptions
        {
            PopularDirectories = popularDirectories,
            UnpopularDirectories = unpopularDirectories,
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
            ValidationFileNames = validationFileNames,
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
            MinRandomCropScale = minRandomCropScale,
            EnableGroupAwareTraining = enableGroupAwareTraining
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

        ValidateDirectorySets(options.PopularDirectories, options.UnpopularDirectories);

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
    [--backbone convnext_large] \
    [--pretrained-weights <path>] \
    [--freeze-backbone-epochs 3] \
    [--enable-augmentation true] \
    [--hflip-prob 0.5] \
    [--max-rotation-deg 12] \
    [--brightness-jitter 0.15] \
    [--contrast-jitter 0.15] \
    [--saturation-jitter 0.15] \
    [--min-random-crop-scale 0.85] \
    [--enable-group-aware-training true] \
    [--epochs 20] \
    [--batch-size 128] \
    [--learning-rate 0.0003] \
    [--fine-tune-learning-rate 0.00005] \
    [--weight-decay 0.0001] \
    [--validation-dir validation.txt] \
    [--validation-split 0.1] \
    [--max-samples-per-class 0] \
    [--seed 42]

Notes:
  --popular-dir should contain only popular images (label=1). It can be
  provided multiple times or with ',' / ';' separated paths.
  --unpopular-dir should contain only unpopular images (label=0). It can be
  provided multiple times or with ',' / ';' separated paths.
  No two provided popular/unpopular directories may be the same directory or
  contain one another.
  --output-model is only a file-name prefix, not a path. It must not contain
  '\' or '/'. Generated models are always written under models/.
  --validation-dir, when provided, is treated as one or more file names to
  look up recursively under both --popular-dir and --unpopular-dir. Pass it
  multiple times or separate names with ',' / ';'. Matching image files are
  used directly as validation data. Matching .txt files are read line by
  line, and each non-empty line is treated as an image path and merged into
  the validation set. Explicit validation image references take priority over
  training if the same image path appears in both. Missing validation file
  names are ignored. The combined explicit validation image count must also
  be at least total-images * validation-split.
  If --validation-dir is omitted, validation data is chosen by random
  stratified split controlled by --validation-split.
  The trainer always auto-generates the rest of the model file name using train
  sample count, image size, augmentation flag, decision threshold, epochs,
  batch size, seed, and training completion time (month/day/hour/minute).
  Group-aware training is enabled by default. When
  --enable-group-aware-training is true, the trainer will infer a group id
  from the numeric prefix before the first '_' in each file name, use it for
  group-aware train/validation separation, down-weight larger near-duplicate
  groups during training, and limit per-batch repeats from the same group.
  Pass --enable-group-aware-training false to disable all group-aware
  behavior and fall back to the original sample-level logic.
  Preprocess cache is always enabled for training.
  Pretrained backbone is always enabled.
  Supported backbones: convnext_tiny / convnext_small / convnext_base /
  convnext_large.
  If --pretrained-weights is omitted, the trainer will auto-download the
  official ConvNeXt ImageNet weights from download.pytorch.org.
""";
    }

    public ImagePopularityTrainingOptions ToCoreOptions()
    {
        return new ImagePopularityTrainingOptions
        {
            PopularDirectories = PopularDirectories,
            UnpopularDirectories = UnpopularDirectories,
            OutputModelPrefix = OutputModelPrefix,
            PreprocessCacheDirectory = PreprocessCacheDirectory,
            TrainImageSize = TrainImageSize,
            Epochs = Epochs,
            BatchSize = BatchSize,
            LearningRate = LearningRate,
            FineTuneLearningRate = FineTuneLearningRate,
            WeightDecay = WeightDecay,
            ValidationFileNames = ValidationFileNames,
            ValidationSplit = ValidationSplit,
            Seed = Seed,
            MaxSamplesPerClass = MaxSamplesPerClass,
            Backbone = Backbone,
            PretrainedWeightsFile = PretrainedWeightsFile,
            FreezeBackboneEpochs = FreezeBackboneEpochs,
            EnableAugmentation = EnableAugmentation,
            HorizontalFlipProbability = HorizontalFlipProbability,
            MaxRotationDegrees = MaxRotationDegrees,
            BrightnessJitter = BrightnessJitter,
            ContrastJitter = ContrastJitter,
            SaturationJitter = SaturationJitter,
            MinRandomCropScale = MinRandomCropScale,
            EnableGroupAwareTraining = EnableGroupAwareTraining
        };
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

    private static IReadOnlyList<string> ParseValidationFileNames(IEnumerable<string> values)
    {
        var fileNames = new List<string>();
        foreach (var value in values)
        {
            if (string.IsNullOrWhiteSpace(value))
            {
                continue;
            }

            var segments = value.Split([',', ';'], StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
            foreach (var segment in segments)
            {
                var fileName = segment.Trim();
                if (string.IsNullOrWhiteSpace(fileName))
                {
                    continue;
                }

                if (fileName.IndexOfAny([Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar]) >= 0)
                {
                    throw new ArgumentException("validation-dir must be a file name only, not a path.");
                }

                fileNames.Add(fileName);
            }
        }

        return fileNames
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }

    private static IReadOnlyList<string> ParseDirectoryList(IEnumerable<string> values, string optionName)
    {
        var directories = new List<string>();
        foreach (var value in values)
        {
            if (string.IsNullOrWhiteSpace(value))
            {
                continue;
            }

            var segments = value.Split([',', ';'], StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
            foreach (var segment in segments)
            {
                var directory = segment.Trim();
                if (string.IsNullOrWhiteSpace(directory))
                {
                    continue;
                }

                directories.Add(Path.GetFullPath(directory));
            }
        }

        var normalized = directories
            .Select(path => path.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();

        if (normalized.Length == 0)
        {
            throw new ArgumentException($"Missing required option --{optionName}\n\n{GetUsage()}");
        }

        return normalized;
    }

    private static void ValidateDirectorySets(IReadOnlyList<string> popularDirectories, IReadOnlyList<string> unpopularDirectories)
    {
        var allDirectories = popularDirectories
            .Select(path => ("popular-dir", path))
            .Concat(unpopularDirectories.Select(path => ("unpopular-dir", path)))
            .ToArray();

        for (var i = 0; i < allDirectories.Length; i++)
        {
            for (var j = i + 1; j < allDirectories.Length; j++)
            {
                var first = allDirectories[i];
                var second = allDirectories[j];

                if (string.Equals(first.path, second.path, StringComparison.OrdinalIgnoreCase))
                {
                    throw new ArgumentException($"{first.Item1} and {second.Item1} must not reference the same directory: {first.path}");
                }

                if (ContainsDirectory(first.path, second.path) || ContainsDirectory(second.path, first.path))
                {
                    throw new ArgumentException($"Data directories must not contain one another: {first.path} <-> {second.path}");
                }
            }
        }
    }

    private static bool ContainsDirectory(string parentCandidate, string childCandidate)
    {
        if (parentCandidate.Length >= childCandidate.Length)
        {
            return false;
        }

        return childCandidate.StartsWith(parentCandidate, StringComparison.OrdinalIgnoreCase) &&
               (childCandidate[parentCandidate.Length] == Path.DirectorySeparatorChar ||
                childCandidate[parentCandidate.Length] == Path.AltDirectorySeparatorChar);
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
