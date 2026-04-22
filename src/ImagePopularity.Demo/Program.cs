using ImagePopularity.Core;

using var logScope = ExecutionLogScope.Start("demo", args);

if (args.Length < 2)
{
    Console.WriteLine("Usage: dotnet run --project src/ImagePopularity.Demo -- <model> <imageDirectory> [maxPredictionCount] [batchSize] [inferenceImageSize] [enablePreprocessCache]");
    return;
}

var model = args[0];
if (string.IsNullOrWhiteSpace(model))
{
    Console.WriteLine("First argument model cannot be empty.");
    return;
}

if (model.Contains('\\') || model.Contains('/'))
{
    Console.WriteLine("First argument must be only a model file name under the models directory, not a path.");
    return;
}

var modelPath = Path.Combine("models", model);
var imageDirectory = args[1];
int? maxPredictionCount = null;
if (args.Length > 2)
{
    if (!int.TryParse(args[2], out var parsedMaxPredictionCount))
    {
        Console.WriteLine("Third argument must be maxPredictionCount (integer).");
        return;
    }

    maxPredictionCount = parsedMaxPredictionCount;
}

var batchSize = 64;
if (args.Length > 3)
{
    if (!int.TryParse(args[3], out batchSize))
    {
        Console.WriteLine("Fourth argument must be batchSize (integer). Device argument has been removed; CUDA is always used.");
        return;
    }
}

int? inferenceImageSize = null;
if (args.Length > 4)
{
    if (!int.TryParse(args[4], out var parsedInferenceImageSize))
    {
        Console.WriteLine("Fifth argument must be inferenceImageSize (integer).");
        return;
    }

    inferenceImageSize = parsedInferenceImageSize;
}

var enablePreprocessCache = false;
if (args.Length > 5)
{
    if (!bool.TryParse(args[5], out enablePreprocessCache))
    {
        Console.WriteLine("Sixth argument must be enablePreprocessCache (true/false).");
        return;
    }
}

if (!File.Exists(modelPath))
{
    Console.WriteLine($"Model not found under models directory: {modelPath}");
    return;
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
    .Take(maxPredictionCount is > 0 ? maxPredictionCount.Value : int.MaxValue)
    .ToList();

if (imagePaths.Count == 0)
{
    Console.WriteLine("No supported images found.");
    return;
}

using var predictor = new ImagePopularityPredictor(modelPath, new ImagePopularityPredictorOptions
{
    InferenceImageSize = inferenceImageSize,
    EnablePreprocessCache = enablePreprocessCache
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
