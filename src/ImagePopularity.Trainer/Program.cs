using ImagePopularity.Core;
using ImagePopularity.Trainer;

using var logScope = ExecutionLogScope.Start("trainer", args);

try
{
    var options = TrainingOptions.Parse(args);

    Console.WriteLine("Start training popularity model...");
    using var trainer = new ModelTrainer(options);
    var cancellationAnnounced = 0;
    ConsoleCancelEventHandler? cancelHandler = null;
    cancelHandler = (_, eventArgs) =>
    {
        eventArgs.Cancel = true;
        trainer.RequestCancellation();

        if (Interlocked.Exchange(ref cancellationAnnounced, 1) == 0)
        {
            Console.WriteLine("Cancellation requested. Trainer will stop at the next safe point and promote the current best autosave model if available.");
        }
    };

    Console.CancelKeyPress += cancelHandler;
    try
    {
        trainer.Run();
    }
    finally
    {
        Console.CancelKeyPress -= cancelHandler;
    }

    Console.WriteLine("Training finished.");
}
catch (ArgumentException ex) when (ex.Message.Contains("Usage:", StringComparison.Ordinal))
{
    Console.WriteLine(ex.Message);
}
catch (OperationCanceledException ex)
{
    Console.WriteLine($"Training canceled: {ex.Message}");
}
catch (Exception ex)
{
    Console.WriteLine($"Training failed: {ex}");
}
