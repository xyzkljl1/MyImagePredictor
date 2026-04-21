using ImagePopularity.Core;

using var logScope = ExecutionLogScope.Start("demo", args);

if (args.Length < 2)
{
    Console.WriteLine("Usage: dotnet run --project src/ImagePopularity.Demo -- <modelPath> <imageDirectory> [batchSize] [inferenceImageSize] [enablePreprocessCache] [preprocessCacheDirectory]");
    return;
}

var modelPath = args[0];
var imageDirectory = args[1];
var batchSize = 128;
if (args.Length > 2)
{
    if (!int.TryParse(args[2], out batchSize))
    {
        Console.WriteLine("Third argument must be batchSize (integer). Device argument has been removed; CUDA is always used.");
        return;
    }
}

int? inferenceImageSize = null;
if (args.Length > 3)
{
    if (!int.TryParse(args[3], out var parsedInferenceImageSize))
    {
        Console.WriteLine("Fourth argument must be inferenceImageSize (integer).");
        return;
    }

    inferenceImageSize = parsedInferenceImageSize;
}

var enablePreprocessCache = false;
if (args.Length > 4)
{
    if (!bool.TryParse(args[4], out enablePreprocessCache))
    {
        Console.WriteLine("Fifth argument must be enablePreprocessCache (true/false).");
        return;
    }
}

string? preprocessCacheDirectory = null;
if (args.Length > 5)
{
    preprocessCacheDirectory = args[5];
    if (string.IsNullOrWhiteSpace(preprocessCacheDirectory))
    {
        Console.WriteLine("Sixth argument preprocessCacheDirectory cannot be empty.");
        return;
    }
}

if (!Directory.Exists(imageDirectory))
{
    Console.WriteLine($"Directory not found: {imageDirectory}");
    return;
}

if (batchSize <= 0)
{
    Console.WriteLine("batchSize must be > 0.");
    return;
}

if (inferenceImageSize is <= 0)
{
    Console.WriteLine("inferenceImageSize must be > 0.");
    return;
}

var imagePaths = Directory
    .EnumerateFiles(imageDirectory, "*", SearchOption.AllDirectories)
    .Where(SupportedImageFiles.IsSupported)
    .OrderBy(path => path)
    .ToList();

if (imagePaths.Count == 0)
{
    Console.WriteLine("No supported images found.");
    return;
}

using var predictor = new ImagePopularityPredictor(modelPath, new ImagePopularityPredictorOptions
{
    InferenceImageSize = inferenceImageSize,
    EnablePreprocessCache = enablePreprocessCache,
    PreprocessCacheDirectory = string.IsNullOrWhiteSpace(preprocessCacheDirectory)
        ? Path.Combine("models", "inference-cache")
        : preprocessCacheDirectory
});

var probabilities = predictor.PredictProbabilities(imagePaths, batchSize);

for (var i = 0; i < imagePaths.Count; i++)
{
    Console.WriteLine($"{Path.GetFileName(imagePaths[i])}\t{probabilities[i]:F6}");
}
