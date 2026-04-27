using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using ModuleContract;

namespace ReportModule;

public sealed class ReportModule : IAppModule
{
    public string Name => "ReportModule";
    public IReadOnlyCollection<string> Dependencies => ["CatalogModule"];

    public void RegisterServices(IServiceCollection services, IConfiguration configuration)
    {
        services.AddSingleton<IReportRenderer, StudyTaskReportRenderer>();
    }

    public Task InitializeAsync(IServiceProvider serviceProvider, CancellationToken cancellationToken)
    {
        var journal = serviceProvider.GetRequiredService<IRuntimeJournal>();
        var renderer = serviceProvider.GetRequiredService<IReportRenderer>();
        journal.Write("ReportModule generated report:");
        journal.Write(renderer.Render());
        return Task.CompletedTask;
    }
}
