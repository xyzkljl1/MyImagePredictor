using System.Text.Json;

namespace ImagePopularity.Core;

public sealed class PopularityModelMetadata
{
    public int TrainingImageSize { get; init; } = 320;

    public int RecommendedInferenceImageSize { get; init; } = 320;

    public string Backbone { get; init; } = PopularityBackboneCatalog.DefaultBackbone;

    public bool UsedPretrainedBackbone { get; init; }

    public string? PretrainedWeightsReference { get; init; }

    public int Epochs { get; init; }

    public int TrainSamples { get; init; }

    public int ValidationSamples { get; init; }

    public string TrainedDevice { get; init; } = "unknown";

    public DateTimeOffset TrainedAtUtc { get; init; } = DateTimeOffset.UtcNow;

    public static string GetMetadataPath(string modelPath)
    {
        return Path.ChangeExtension(modelPath, ".meta.json");
    }

    public static PopularityModelMetadata? TryLoad(string modelPath)
    {
        var metadataPath = GetMetadataPath(modelPath);
        if (!File.Exists(metadataPath))
        {
            return null;
        }

        var json = File.ReadAllText(metadataPath);
        return JsonSerializer.Deserialize<PopularityModelMetadata>(json);
    }

    public void Save(string modelPath)
    {
        var metadataPath = GetMetadataPath(modelPath);
        var json = JsonSerializer.Serialize(this, new JsonSerializerOptions
        {
            WriteIndented = true
        });

        File.WriteAllText(metadataPath, json);
    }
}
