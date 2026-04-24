namespace ImagePopularity.Core;

public static class PopularityBackboneCatalog
{
    private const string DefaultBackboneName = "convnext_large";

    private static readonly IReadOnlyDictionary<string, BackboneSpec> Specs =
        new Dictionary<string, BackboneSpec>(StringComparer.Ordinal)
        {
            ["convnexttiny"] = new(
                DisplayName: "convnext_tiny",
                FeatureDimension: ConvNeXtFactory.GetFeatureDimension("convnexttiny"),
                DefaultWeightsFileName: ConvNeXtFactory.GetDefaultWeightsFileName("convnexttiny")),
            ["convnextsmall"] = new(
                DisplayName: "convnext_small",
                FeatureDimension: ConvNeXtFactory.GetFeatureDimension("convnextsmall"),
                DefaultWeightsFileName: ConvNeXtFactory.GetDefaultWeightsFileName("convnextsmall")),
            ["convnextbase"] = new(
                DisplayName: "convnext_base",
                FeatureDimension: ConvNeXtFactory.GetFeatureDimension("convnextbase"),
                DefaultWeightsFileName: ConvNeXtFactory.GetDefaultWeightsFileName("convnextbase")),
            ["convnextlarge"] = new(
                DisplayName: "convnext_large",
                FeatureDimension: ConvNeXtFactory.GetFeatureDimension("convnextlarge"),
                DefaultWeightsFileName: ConvNeXtFactory.GetDefaultWeightsFileName("convnextlarge"))
        };

    public static string DefaultBackbone => DefaultBackboneName;

    public static string SupportedList => string.Join("/", Specs.Values.Select(spec => spec.DisplayName));

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

    public static TorchSharp.torch.nn.Module<TorchSharp.torch.Tensor, TorchSharp.torch.Tensor> CreateBackbone(
        string backboneName,
        string? weightsFile,
        TorchSharp.torch.Device device)
    {
        var normalized = Normalize(backboneName);
        GetSpec(normalized);
        return ConvNeXtFactory.CreateBackbone(normalized, weightsFile, device);
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
        string DisplayName,
        int FeatureDimension,
        string DefaultWeightsFileName);
}
