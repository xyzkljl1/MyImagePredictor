using System.Text;

namespace ImagePopularity.Core;

public sealed class ExecutionLogScope : IDisposable
{
    private readonly TextWriter? _originalOut;
    private readonly TextWriter? _originalError;
    private readonly StreamWriter? _fileWriter;
    private readonly TeeTextWriter? _teeOut;
    private readonly TeeTextWriter? _teeError;
    private bool _disposed;

    private ExecutionLogScope(
        TextWriter? originalOut,
        TextWriter? originalError,
        StreamWriter? fileWriter,
        TeeTextWriter? teeOut,
        TeeTextWriter? teeError)
    {
        _originalOut = originalOut;
        _originalError = originalError;
        _fileWriter = fileWriter;
        _teeOut = teeOut;
        _teeError = teeError;
    }

    public static ExecutionLogScope Start(string applicationName, IReadOnlyList<string> args)
    {
        try
        {
            var safeName = string.IsNullOrWhiteSpace(applicationName)
                ? "app"
                : SanitizeFileName(applicationName);
            var timestamp = DateTimeOffset.Now.ToString("yyyyMMdd-HHmmss-fff");
            var logDirectory = Path.Combine(Directory.GetCurrentDirectory(), "log");
            Directory.CreateDirectory(logDirectory);

            var logPath = Path.Combine(logDirectory, $"{safeName}-{timestamp}.log");
            var originalOut = Console.Out;
            var originalError = Console.Error;
            var fileWriter = new StreamWriter(logPath, append: false, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false))
            {
                AutoFlush = true
            };

            WriteHeader(fileWriter, safeName, logPath, args);

            var teeOut = new TeeTextWriter(originalOut, fileWriter);
            var teeError = new TeeTextWriter(originalError, fileWriter);
            Console.SetOut(teeOut);
            Console.SetError(teeError);

            return new ExecutionLogScope(originalOut, originalError, fileWriter, teeOut, teeError);
        }
        catch
        {
            return new ExecutionLogScope(null, null, null, null, null);
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        try
        {
            if (_originalOut is not null)
            {
                Console.SetOut(_originalOut);
            }

            if (_originalError is not null)
            {
                Console.SetError(_originalError);
            }
        }
        finally
        {
            _teeOut?.Dispose();
            _teeError?.Dispose();
            _fileWriter?.Dispose();
            _disposed = true;
        }
    }

    private static void WriteHeader(StreamWriter writer, string applicationName, string logPath, IReadOnlyList<string> args)
    {
        writer.WriteLine($"Application: {applicationName}");
        writer.WriteLine($"StartedAtLocal: {DateTimeOffset.Now:yyyy-MM-dd HH:mm:ss zzz}");
        writer.WriteLine($"WorkingDirectory: {Directory.GetCurrentDirectory()}");
        writer.WriteLine($"LogFile: {logPath}");
        writer.WriteLine($"RawCommandLine: {Environment.CommandLine}");
        writer.WriteLine($"ArgumentCount: {args.Count}");

        for (var i = 0; i < args.Count; i++)
        {
            writer.WriteLine($"Arg[{i}]: {EscapeForLog(args[i])}");
        }

        writer.WriteLine(new string('-', 80));
    }

    private static string SanitizeFileName(string value)
    {
        var invalid = Path.GetInvalidFileNameChars();
        var builder = new StringBuilder(value.Length);
        foreach (var ch in value)
        {
            builder.Append(invalid.Contains(ch) ? '_' : ch);
        }

        return builder.ToString();
    }

    private static string EscapeForLog(string value)
    {
        return value
            .Replace("\r", "\\r", StringComparison.Ordinal)
            .Replace("\n", "\\n", StringComparison.Ordinal);
    }

    private sealed class TeeTextWriter : TextWriter
    {
        private readonly TextWriter _consoleWriter;
        private readonly TextWriter _fileWriter;
        private readonly object _syncRoot = new();
        private readonly StringBuilder _currentFileLine = new();

        public TeeTextWriter(TextWriter consoleWriter, TextWriter fileWriter)
        {
            _consoleWriter = consoleWriter;
            _fileWriter = fileWriter;
        }

        public override Encoding Encoding => _consoleWriter.Encoding;

        public override void Write(char value)
        {
            lock (_syncRoot)
            {
                _consoleWriter.Write(value);
                AppendToFileLineBuffer(value);
            }
        }

        public override void Write(string? value)
        {
            if (value is null)
            {
                return;
            }

            lock (_syncRoot)
            {
                _consoleWriter.Write(value);
                AppendToFileLineBuffer(value);
            }
        }

        public override void WriteLine()
        {
            lock (_syncRoot)
            {
                _consoleWriter.WriteLine();
                FlushCurrentFileLine(writeEmptyLineIfBufferEmpty: true);
            }
        }

        public override void WriteLine(string? value)
        {
            lock (_syncRoot)
            {
                _consoleWriter.WriteLine(value);
                AppendToFileLineBuffer(value);
                FlushCurrentFileLine(writeEmptyLineIfBufferEmpty: true);
            }
        }

        public override void Flush()
        {
            lock (_syncRoot)
            {
                _consoleWriter.Flush();
                _fileWriter.Flush();
            }
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                lock (_syncRoot)
                {
                    FlushCurrentFileLine(writeEmptyLineIfBufferEmpty: false);
                    _consoleWriter.Flush();
                    _fileWriter.Flush();
                }
            }

            base.Dispose(disposing);
        }

        private void AppendToFileLineBuffer(string? value)
        {
            if (string.IsNullOrEmpty(value))
            {
                return;
            }

            for (var i = 0; i < value.Length; i++)
            {
                var ch = value[i];
                if (ch == '\r')
                {
                    // Treat CRLF inside a single write as a normal newline so exception
                    // messages and other multiline strings are preserved in the log.
                    if (i + 1 < value.Length && value[i + 1] == '\n')
                    {
                        FlushCurrentFileLine(writeEmptyLineIfBufferEmpty: true);
                        i++;
                        continue;
                    }

                    // A lone carriage return is still used by progress bars to overwrite
                    // the current console line, so keep the compact log behavior there.
                    _currentFileLine.Clear();
                    continue;
                }

                if (ch == '\n')
                {
                    FlushCurrentFileLine(writeEmptyLineIfBufferEmpty: true);
                    continue;
                }

                _currentFileLine.Append(ch);
            }
        }

        private void AppendToFileLineBuffer(char value)
        {
            if (value == '\r')
            {
                _currentFileLine.Clear();
                return;
            }

            if (value == '\n')
            {
                FlushCurrentFileLine(writeEmptyLineIfBufferEmpty: true);
                return;
            }

            _currentFileLine.Append(value);
        }

        private void FlushCurrentFileLine(bool writeEmptyLineIfBufferEmpty)
        {
            if (_currentFileLine.Length == 0)
            {
                if (writeEmptyLineIfBufferEmpty)
                {
                    _fileWriter.WriteLine();
                }

                return;
            }

            _fileWriter.WriteLine(_currentFileLine.ToString());
            _currentFileLine.Clear();
        }
    }
}
