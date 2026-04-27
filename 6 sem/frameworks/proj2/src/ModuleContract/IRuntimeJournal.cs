namespace ModuleContract;

public interface IRuntimeJournal
{
    IReadOnlyList<string> Entries { get; }
    void Write(string message);
}
