using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using ModuleContract;
using proj2.Core;
using proj2.Core.Exceptions;

namespace Proj2.Tests;

public sealed class ModuleDependencyResolverTests
{
    private readonly ModuleDependencyResolver _resolver = new();

    [Fact]
    public void Resolve_ShouldReturnModulesInDependencyOrder()
    {
        var modules = new IAppModule[]
        {
            new FakeModule("Export", ["Report"]),
            new FakeModule("Catalog", []),
            new FakeModule("Report", ["Catalog"])
        };

        var ordered = _resolver.Resolve(["Export", "Report", "Catalog"], modules);

        Assert.Equal(["Catalog", "Report", "Export"], ordered.Select(x => x.Name).ToArray());
    }

    [Fact]
    public void Resolve_ShouldThrowClearMessage_WhenModuleIsMissing()
    {
        var modules = new IAppModule[]
        {
            new FakeModule("Catalog", [])
        };

        var exception = Assert.Throws<MissingModuleException>(() => _resolver.Resolve(["Catalog", "Export"], modules));

        Assert.Equal("Module 'Export' is required but was not found.", exception.Message);
    }

    [Fact]
    public void Resolve_ShouldThrowClearMessage_WhenCycleExists()
    {
        var modules = new IAppModule[]
        {
            new FakeModule("Alpha", ["Beta"]),
            new FakeModule("Beta", ["Alpha"])
        };

        var exception = Assert.Throws<CircularDependencyException>(() => _resolver.Resolve(["Alpha", "Beta"], modules));

        Assert.Equal("Circular dependency detected: Alpha -> Beta -> Alpha", exception.Message);
    }

    [Fact]
    public void RealModules_ShouldRegisterDependenciesThroughContainer()
    {
        var services = new ServiceCollection();
        services.AddSingleton<IRuntimeJournal, RuntimeJournal>();

        var configuration = new ConfigurationBuilder()
            .AddInMemoryCollection(new Dictionary<string, string?>
            {
                ["Export:OutputPath"] = Path.Combine(Path.GetTempPath(), "proj2-tests", "report.txt")
            })
            .Build();

        var modulesPath = GetModulesPath();
        var loader = new ReflectionModuleLoader();
        var modules = loader.LoadModules(modulesPath);
        var ordered = _resolver.Resolve(["CatalogModule", "ReportModule"], modules);

        foreach (var module in ordered)
        {
            module.RegisterServices(services, configuration);
        }

        using var provider = services.BuildServiceProvider();
        var renderer = provider.GetRequiredService<IReportRenderer>();

        var report = renderer.Render();

        Assert.Contains("Prepare module demo", report);
        Assert.Contains("Build middleware prototype", report);
    }

    private static string GetModulesPath()
    {
        return Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../../modules/net8.0"));
    }

    private sealed class FakeModule(string name, IReadOnlyCollection<string> dependencies) : IAppModule
    {
        public string Name => name;
        public IReadOnlyCollection<string> Dependencies => dependencies;

        public void RegisterServices(IServiceCollection services, IConfiguration configuration)
        {
        }

        public Task InitializeAsync(IServiceProvider serviceProvider, CancellationToken cancellationToken)
        {
            return Task.CompletedTask;
        }
    }
}
