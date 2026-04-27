using System.ComponentModel.DataAnnotations;

namespace proj3.Configuration;

public sealed class RateLimitingOptions
{
    [Range(1, int.MaxValue)]
    public int PermitLimit { get; init; } = 3;

    [Range(1, int.MaxValue)]
    public int WindowSeconds { get; init; } = 10;
}
