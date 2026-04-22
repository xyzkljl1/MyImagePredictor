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

var probabilities = new List<float>(imagePaths.Count);
using var progress = new ConsoleProgressBar("Predict", imagePaths.Count);

for (var start = 0; start < imagePaths.Count; start += batchSize)
{
    var currentBatch = Math.Min(batchSize, imagePaths.Count - start);
    var batchImagePaths = imagePaths
        .Skip(start)
        .Take(currentBatch)
        .ToArray();
    var batchProbabilities = predictor.PredictProbabilities(batchImagePaths, currentBatch);
    probabilities.AddRange(batchProbabilities);
    progress.Report(start + currentBatch);
}

progress.Complete();

double probabilitySum = 0;
var aboveThresholdCount = 0;

for (var i = 0; i < imagePaths.Count; i++)
{
    var probability = probabilities[i];
    probabilitySum += probability;

    if (probability > 0.5f)
    {
        aboveThresholdCount++;
        Console.WriteLine($"{Path.GetFileName(imagePaths[i])}\t{probability:F6}");
    }
}

var averageProbability = probabilitySum / imagePaths.Count;
Console.WriteLine($"Average probability:\t{averageProbability:F6}");
Console.WriteLine($"> 0.5 count:\t{aboveThresholdCount}");
Console.WriteLine($"Total images:\t{imagePaths.Count}");
