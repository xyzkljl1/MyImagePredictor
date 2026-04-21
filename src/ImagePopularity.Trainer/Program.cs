using ImagePopularity.Core;
using ImagePopularity.Trainer;

using var logScope = ExecutionLogScope.Start("trainer", args);

try
{
    var options = TrainingOptions.Parse(args);

    Console.WriteLine("Start training popularity model...");
    using var trainer = new ModelTrainer(options);
    trainer.Run();
    Console.WriteLine("Training finished.");
}
catch (ArgumentException ex) when (ex.Message.Contains("Usage:", StringComparison.Ordinal))
{
    Console.WriteLine(ex.Message);
}
catch (Exception ex)
{
    Console.WriteLine($"Training failed: {ex}");
}
