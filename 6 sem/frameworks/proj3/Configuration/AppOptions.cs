using System.ComponentModel.DataAnnotations;

namespace proj3.Configuration;

public sealed class AppOptions
{
    public const string SectionName = "App";

    [Required]
    public RuntimeMode Mode { get; init; } = RuntimeMode.Study;

    public string[] AllowedOrigins { get; init; } = [];

    [Required]
    public RateLimitingOptions RateLimiting { get; init; } = new();
}
