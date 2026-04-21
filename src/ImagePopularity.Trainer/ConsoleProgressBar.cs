using System.Diagnostics;
using System.Globalization;

namespace ImagePopularity.Trainer;

internal sealed class ConsoleProgressBar : IDisposable
{
    private readonly string _label;
    private readonly int _total;
    private readonly int _barWidth;
    private readonly Stopwatch _elapsed = Stopwatch.StartNew();
    private readonly Stopwatch _throttle = Stopwatch.StartNew();

    private int _lastPrintedLength;
    private int _lastRenderedCurrent = -1;
    private bool _completed;

    public ConsoleProgressBar(string label, int total, int barWidth = 28)
    {
        _label = string.IsNullOrWhiteSpace(label) ? "Progress" : label;
        _total = Math.Max(total, 0);
        _barWidth = Math.Clamp(barWidth, 10, 80);

        Render(0, status: null, force: true);
    }

    public void Report(int current, string? status = null)
    {
        if (_completed)
        {
            return;
        }

        var normalizedCurrent = _total == 0
            ? 0
            : Math.Clamp(current, 0, _total);

        if (normalizedCurrent == _lastRenderedCurrent && normalizedCurrent != _total)
        {
            return;
        }

        if (normalizedCurrent != _total && _throttle.ElapsedMilliseconds < 100)
        {
            return;
        }

        Render(normalizedCurrent, status, force: normalizedCurrent == _total);
    }

    public void Complete(string? status = null)
    {
        if (_completed)
        {
            return;
        }

        var finalCurrent = _total == 0 ? 0 : _total;
        Render(finalCurrent, status, force: true);
        Console.WriteLine();
        _completed = true;
    }

    public void Dispose()
    {
        if (!_completed)
        {
            Complete();
        }
    }

    private void Render(int current, string? status, bool force)
    {
        var ratio = _total == 0 ? 1.0 : current / (double)_total;
        ratio = Math.Clamp(ratio, 0.0, 1.0);

        var filled = (int)Math.Round(ratio * _barWidth, MidpointRounding.AwayFromZero);
        filled = Math.Clamp(filled, 0, _barWidth);

        var bar = string.Concat(
            new string('#', filled),
            new string('-', _barWidth - filled));
        var percentText = (ratio * 100).ToString("0.00", CultureInfo.InvariantCulture);
        var elapsedText = FormatElapsed(_elapsed.Elapsed);

        var text = $"\r{_label} [{bar}] {percentText}% ({current}/{_total}) | elapsed={elapsedText}";
        if (!string.IsNullOrWhiteSpace(status))
        {
            text += $" | {status}";
        }

        if (text.Length < _lastPrintedLength)
        {
            text += new string(' ', _lastPrintedLength - text.Length);
        }

        Console.Write(text);

        _lastPrintedLength = text.Length;
        _lastRenderedCurrent = current;

        if (force)
        {
            _throttle.Restart();
        }
    }

    private static string FormatElapsed(TimeSpan elapsed)
    {
        if (elapsed.TotalHours >= 1)
        {
            return elapsed.ToString(@"hh\:mm\:ss", CultureInfo.InvariantCulture);
        }

        return elapsed.ToString(@"mm\:ss", CultureInfo.InvariantCulture);
    }
}
