using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using ModuleContract;

namespace CycleAlphaModule;

public sealed class CycleAlphaModule : IAppModule
{
    public string Name => "CycleAlphaModule";
    public IReadOnlyCollection<string> Dependencies => ["CycleBetaModule"];

    public void RegisterServices(IServiceCollection services, IConfiguration configuration)
    {
    }

    public Task InitializeAsync(IServiceProvider serviceProvider, CancellationToken cancellationToken)
    {
        return Task.CompletedTask;
    }
}
