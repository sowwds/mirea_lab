using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using ModuleContract;
using proj2.Core;
using proj2.Core.Configuration;

try
{
    var configFile = CommandLineConfigResolver.ResolveConfigFile(args);

    var builder = Host.CreateApplicationBuilder(args);
    builder.Configuration.Sources.Clear();
    builder.Configuration
        .AddJsonFile(configFile, optional: false, reloadOnChange: false)
        .AddEnvironmentVariables()
        .AddCommandLine(args);

    builder.Services.AddSingleton<IRuntimeJournal, RuntimeJournal>();
    builder.Services.AddSingleton<IModuleLoader, ReflectionModuleLoader>();
    builder.Services.AddSingleton<IModuleDependencyResolver, ModuleDependencyResolver>();

    var moduleOptions = builder.Configuration
        .GetSection(ModuleOptions.SectionName)
        .Get<ModuleOptions>() ?? new ModuleOptions();

    var modulesPath = ModulePathResolver.Resolve(builder.Environment.ContentRootPath, moduleOptions.ModulesPath);
    var loader = new ReflectionModuleLoader();
    var availableModules = loader.LoadModules(modulesPath);
    var orderedModules = new ModuleDependencyResolver().Resolve(moduleOptions.EnabledModules, availableModules);

    foreach (var module in orderedModules)
    {
        module.RegisterServices(builder.Services, builder.Configuration);
    }

    using var host = builder.Build();
    var journal = host.Services.GetRequiredService<IRuntimeJournal>();

    journal.Write($"Loaded modules from {modulesPath}");
    journal.Write($"Initialization order: {string.Join(" -> ", orderedModules.Select(x => x.Name))}");

    foreach (var module in orderedModules)
    {
        journal.Write($"Initializing module: {module.Name}");
        await module.InitializeAsync(host.Services, CancellationToken.None);
    }

    foreach (var entry in journal.Entries)
    {
        Console.WriteLine(entry);
    }
}
catch (Exception exception)
{
    Console.Error.WriteLine($"Startup failed: {exception.Message}");
    Environment.ExitCode = 1;
}
