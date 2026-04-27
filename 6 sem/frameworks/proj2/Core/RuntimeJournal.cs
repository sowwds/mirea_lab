using ModuleContract;

namespace proj2.Core;

public sealed class RuntimeJournal : IRuntimeJournal
{
    private readonly List<string> _entries = [];

    public IReadOnlyList<string> Entries => _entries;

    public void Write(string message)
    {
        _entries.Add(message);
    }
}
