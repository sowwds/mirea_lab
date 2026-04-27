using Microsoft.Extensions.Configuration;
using proj3.Configuration;

namespace Proj3.Tests;

public sealed class ConfigurationTests
{
    [Fact]
    public void ConfigurationPriority_ShouldBe_JsonThenEnvironmentThenCommandLine()
    {
        var environmentLikeOverrides = new Dictionary<string, string?>
        {
            ["App:Mode"] = "Production",
            ["App:RateLimiting:PermitLimit"] = "5"
        };

        var commandLine = new[]
        {
            "--App:RateLimiting:PermitLimit=9"
        };

        var configuration = new ConfigurationBuilder()
            .SetBasePath(GetProjectRoot())
            .AddJsonFile("appsettings.json", optional: false)
            .AddInMemoryCollection(environmentLikeOverrides)
            .AddCommandLine(commandLine)
            .Build();

        var options = configuration.GetSection(AppOptions.SectionName).Get<AppOptions>();

        Assert.NotNull(options);
        Assert.Equal(RuntimeMode.Production, options!.Mode);
        Assert.Equal(9, options.RateLimiting.PermitLimit);
        Assert.Equal(10, options.RateLimiting.WindowSeconds);
    }

    [Fact]
    public void InvalidConfiguration_ShouldExposeBrokenValues()
    {
        var configuration = new ConfigurationBuilder()
            .SetBasePath(GetProjectRoot())
            .AddJsonFile("appsettings.invalid.json", optional: false)
            .Build();

        var options = configuration.GetSection(AppOptions.SectionName).Get<AppOptions>();

        Assert.NotNull(options);
        Assert.Empty(options!.AllowedOrigins);
        Assert.Equal(0, options.RateLimiting.PermitLimit);
        Assert.Equal(0, options.RateLimiting.WindowSeconds);
    }

    private static string GetProjectRoot()
    {
        return Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../../"));
    }
}
