using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using ModuleContract;

namespace ExportModule;

public sealed class ExportModule : IAppModule
{
    public string Name => "ExportModule";
    public IReadOnlyCollection<string> Dependencies => ["ReportModule"];

    public void RegisterServices(IServiceCollection services, IConfiguration configuration)
    {
    }

    public async Task InitializeAsync(IServiceProvider serviceProvider, CancellationToken cancellationToken)
    {
        var configuration = serviceProvider.GetRequiredService<IConfiguration>();
        var journal = serviceProvider.GetRequiredService<IRuntimeJournal>();
        var renderer = serviceProvider.GetRequiredService<IReportRenderer>();

        var outputPath = configuration["Export:OutputPath"] ?? "output/report.txt";
        var fullPath = Path.GetFullPath(outputPath);
        Directory.CreateDirectory(Path.GetDirectoryName(fullPath)!);

        await File.WriteAllTextAsync(fullPath, renderer.Render(), cancellationToken);
        journal.Write($"ExportModule wrote report to {fullPath}");
    }
}
