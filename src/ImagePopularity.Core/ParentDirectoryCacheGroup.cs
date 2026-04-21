using System.Text;

namespace ImagePopularity.Core;

internal readonly record struct ParentDirectoryCacheGroup(string LogicalName, string SafeName)
{
    public static ParentDirectoryCacheGroup FromSourcePath(string sourceImagePath)
    {
        if (string.IsNullOrWhiteSpace(sourceImagePath))
        {
            throw new ArgumentException("Source path cannot be empty.", nameof(sourceImagePath));
        }

        var fullPath = Path.GetFullPath(sourceImagePath);
        var parentDirectoryPath = Path.GetDirectoryName(fullPath);

        var logicalName = "_root";
        if (!string.IsNullOrWhiteSpace(parentDirectoryPath))
        {
            var parentDirectory = new DirectoryInfo(parentDirectoryPath);
            if (!string.IsNullOrWhiteSpace(parentDirectory.Name))
            {
                logicalName = parentDirectory.Name;
            }
        }

        return new ParentDirectoryCacheGroup(logicalName, ToSafeName(logicalName));
    }

    private static string ToSafeName(string logicalName)
    {
        var builder = new StringBuilder(logicalName.Length);
        var invalidChars = Path.GetInvalidFileNameChars();

        foreach (var ch in logicalName)
        {
            if (char.IsControl(ch) || invalidChars.Contains(ch))
            {
                builder.Append('_');
            }
            else
            {
                builder.Append(ch);
            }
        }

        var safeName = builder
            .ToString()
            .Trim()
            .TrimEnd('.');

        return string.IsNullOrWhiteSpace(safeName) ? "_root" : safeName;
    }
}
