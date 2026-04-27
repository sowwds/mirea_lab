namespace proj4.Configuration;

public sealed class OperationsOptions
{
    public const string SectionName = "Operations";

    public int CriticalFailureThreshold { get; init; } = 2;
}
