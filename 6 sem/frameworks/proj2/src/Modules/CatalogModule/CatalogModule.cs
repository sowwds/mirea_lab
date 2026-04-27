using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using ModuleContract;

namespace CatalogModule;

public sealed class CatalogModule : IAppModule
{
    public string Name => "CatalogModule";
    public IReadOnlyCollection<string> Dependencies => [];

    public void RegisterServices(IServiceCollection services, IConfiguration configuration)
    {
        services.AddSingleton<IStudyTaskCatalog, InMemoryStudyTaskCatalog>();
    }

    public Task InitializeAsync(IServiceProvider serviceProvider, CancellationToken cancellationToken)
    {
        var journal = serviceProvider.GetRequiredService<IRuntimeJournal>();
        var catalog = serviceProvider.GetRequiredService<IStudyTaskCatalog>();
        journal.Write($"CatalogModule registered {catalog.GetAll().Count} tasks.");
        return Task.CompletedTask;
    }
}
