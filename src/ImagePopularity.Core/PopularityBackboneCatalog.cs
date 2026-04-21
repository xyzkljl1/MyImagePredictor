using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace ImagePopularity.Core;

public static class PopularityBackboneCatalog
{
    private const string DefaultBackboneName = "resnet152";

    private static readonly IReadOnlyDictionary<string, BackboneSpec> Specs =
        new Dictionary<string, BackboneSpec>(StringComparer.Ordinal)
        {
            ["resnet18"] = new(
                FeatureDimension: 512,
                DefaultWeightsFileName: "ResNet18_Weights.IMAGENET1K_V1",
                Factory: (weightsFile, device) => TorchSharp.torchvision.models.resnet18(weights_file: weightsFile, skipfc: true, device: device)),
            ["resnet34"] = new(
                FeatureDimension: 512,
                DefaultWeightsFileName: "ResNet34_Weights.IMAGENET1K_V1",
                Factory: (weightsFile, device) => TorchSharp.torchvision.models.resnet34(weights_file: weightsFile, skipfc: true, device: device)),
            ["resnet50"] = new(
                FeatureDimension: 2048,
                DefaultWeightsFileName: "ResNet50_Weights.IMAGENET1K_V2",
                Factory: (weightsFile, device) => TorchSharp.torchvision.models.resnet50(weights_file: weightsFile, skipfc: true, device: device)),
            ["resnet101"] = new(
                FeatureDimension: 2048,
                DefaultWeightsFileName: "ResNet101_Weights.IMAGENET1K_V2",
                Factory: (weightsFile, device) => TorchSharp.torchvision.models.resnet101(weights_file: weightsFile, skipfc: true, device: device)),
            ["resnet152"] = new(
                FeatureDimension: 2048,
                DefaultWeightsFileName: "ResNet152_Weights.IMAGENET1K_V2",
                Factory: (weightsFile, device) => TorchSharp.torchvision.models.resnet152(weights_file: weightsFile, skipfc: true, device: device))
        };

    public static string DefaultBackbone => DefaultBackboneName;

    public static string SupportedList => string.Join("/", Specs.Keys);

    public static string Normalize(string? backbone)
    {
        if (string.IsNullOrWhiteSpace(backbone))
        {
            return DefaultBackboneName;
        }

        return backbone
            .Trim()
            .ToLowerInvariant()
            .Replace("-", string.Empty, StringComparison.Ordinal)
            .Replace("_", string.Empty, StringComparison.Ordinal)
            .Replace(" ", string.Empty, StringComparison.Ordinal);
    }

    public static bool IsSupported(string? backbone)
    {
        return Specs.ContainsKey(Normalize(backbone));
    }

    public static Module<Tensor, Tensor> CreateBackbone(string backboneName, string? weightsFile, Device device)
    {
        return GetSpec(backboneName).Factory(weightsFile, device);
    }

    public static int GetFeatureDimension(string backboneName)
    {
        return GetSpec(backboneName).FeatureDimension;
    }

    public static string GetDefaultPretrainedWeightsFileName(string backboneName)
    {
        return GetSpec(backboneName).DefaultWeightsFileName;
    }

    private static BackboneSpec GetSpec(string backboneName)
    {
        var normalized = Normalize(backboneName);
        if (Specs.TryGetValue(normalized, out var spec))
        {
            return spec;
        }

        throw new ArgumentOutOfRangeException(nameof(backboneName), backboneName, $"Unsupported backbone. Use {SupportedList}.");
    }

    private sealed record BackboneSpec(
        int FeatureDimension,
        string DefaultWeightsFileName,
        Func<string?, Device, Module<Tensor, Tensor>> Factory);
}
