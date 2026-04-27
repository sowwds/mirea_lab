using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;

namespace ModuleContract;

public interface IAppModule
{
    string Name { get; }
    IReadOnlyCollection<string> Dependencies { get; }
    void RegisterServices(IServiceCollection services, IConfiguration configuration);
    Task InitializeAsync(IServiceProvider serviceProvider, CancellationToken cancellationToken);
}
