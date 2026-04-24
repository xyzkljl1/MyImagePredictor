using System.Net.Http;

namespace ImagePopularity.Core;

internal static class PretrainedWeightsResolver
{
    private const string BaseUrl = "https://download.pytorch.org/models";
    private static readonly string DefaultCacheDirectory = Path.Combine("models", "pretrained");

    private static readonly HttpClient HttpClient = new()
    {
        Timeout = TimeSpan.FromMinutes(30)
    };

    public static string Resolve(ImagePopularityTrainingOptions options)
    {
        if (!string.IsNullOrWhiteSpace(options.PretrainedWeightsFile))
        {
            var explicitWeightsPath = Path.GetFullPath(options.PretrainedWeightsFile);
            if (!File.Exists(explicitWeightsPath))
            {
                throw new FileNotFoundException($"Pretrained weights file not found: {explicitWeightsPath}", explicitWeightsPath);
            }

            return explicitWeightsPath;
        }

        var fileName = PopularityBackboneCatalog.GetDefaultPretrainedWeightsFileName(options.Backbone);

        var cacheDirectory = Path.GetFullPath(DefaultCacheDirectory);
        Directory.CreateDirectory(cacheDirectory);

        var targetPath = Path.Combine(cacheDirectory, fileName);
        if (File.Exists(targetPath))
        {
            Console.WriteLine($"Using cached pretrained weights: {targetPath}");
            return targetPath;
        }

        var url = $"{BaseUrl}/{fileName}";
        Console.WriteLine($"Downloading official pretrained weights for {options.Backbone} from: {url}");
        DownloadFile(url, targetPath).GetAwaiter().GetResult();
        Console.WriteLine($"Pretrained weights downloaded: {targetPath}");

        return targetPath;
    }

    private static async Task DownloadFile(string url, string targetPath)
    {
        using var response = await HttpClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead);
        response.EnsureSuccessStatusCode();

        await using var source = await response.Content.ReadAsStreamAsync();
        await using var destination = File.Create(targetPath);
        await source.CopyToAsync(destination);
    }
}
