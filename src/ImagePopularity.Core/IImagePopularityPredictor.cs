namespace ImagePopularity.Core;

public interface IImagePopularityPredictor : IDisposable
{
    float PredictProbability(string imagePath);

    IReadOnlyList<float> PredictProbabilities(IReadOnlyList<string> imagePaths, int batchSize = 64);
}
