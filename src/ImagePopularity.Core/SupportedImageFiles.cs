namespace ImagePopularity.Core;

public static class SupportedImageFiles
{
    // Keep README's supported-format section in sync with this list.
    private static readonly HashSet<string> ExtensionsSet = new(StringComparer.OrdinalIgnoreCase)
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".gif",
        ".webp",
        ".tif",
        ".tiff"
    };

    public static IReadOnlySet<string> Extensions => ExtensionsSet;

    public static bool IsSupported(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        return ExtensionsSet.Contains(Path.GetExtension(path));
    }
}
