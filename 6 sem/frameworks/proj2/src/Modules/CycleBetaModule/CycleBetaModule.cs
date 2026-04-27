using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using ModuleContract;

namespace CycleBetaModule;

public sealed class CycleBetaModule : IAppModule
{
    public string Name => "CycleBetaModule";
    public IReadOnlyCollection<string> Dependencies => ["CycleAlphaModule"];

    public void RegisterServices(IServiceCollection services, IConfiguration configuration)
    {
    }

    public Task InitializeAsync(IServiceProvider serviceProvider, CancellationToken cancellationToken)
    {
        return Task.CompletedTask;
    }
}
