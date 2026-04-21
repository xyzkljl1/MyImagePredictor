namespace ImagePopularity.Core;

public sealed class PopularityModelConfig
{
    public string Backbone { get; init; } = PopularityBackboneCatalog.DefaultBackbone;

    public static string NormalizeBackbone(string? backbone)
    {
        return PopularityBackboneCatalog.Normalize(backbone);
    }

    public static bool IsSupportedBackbone(string? backbone)
    {
        return PopularityBackboneCatalog.IsSupported(backbone);
    }
}
