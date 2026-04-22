using ImagePopularity.Core;

namespace ImagePopularity.Trainer;

internal sealed class ModelTrainer : IDisposable
{
    private readonly ImagePopularityTrainer _trainer;

    public ModelTrainer(TrainingOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        _trainer = new ImagePopularityTrainer(options.ToCoreOptions());
    }

    public void Run()
    {
        _trainer.Run();
    }

    public void RequestCancellation()
    {
        _trainer.RequestCancellation();
    }

    public void Dispose()
    {
        _trainer.Dispose();
    }
}
