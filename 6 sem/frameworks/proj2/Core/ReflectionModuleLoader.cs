using System.Reflection;
using System.Runtime.Loader;
using ModuleContract;
using proj2.Core.Exceptions;

namespace proj2.Core;

public sealed class ReflectionModuleLoader : IModuleLoader
{
    public IReadOnlyCollection<IAppModule> LoadModules(string modulesPath)
    {
        if (!Directory.Exists(modulesPath))
        {
            throw new ModuleLoadingException($"Modules directory '{modulesPath}' does not exist.");
        }

        var modules = new List<IAppModule>();
        var moduleType = typeof(IAppModule);

        foreach (var dllPath in Directory.GetFiles(modulesPath, "*.dll", SearchOption.TopDirectoryOnly))
        {
            var assembly = AssemblyLoadContext.Default.LoadFromAssemblyPath(Path.GetFullPath(dllPath));
            var moduleImplementations = assembly
                .GetTypes()
                .Where(type => !type.IsAbstract && !type.IsInterface && moduleType.IsAssignableFrom(type));

            foreach (var implementation in moduleImplementations)
            {
                if (Activator.CreateInstance(implementation) is not IAppModule module)
                {
                    throw new ModuleLoadingException($"Could not create module '{implementation.FullName}'.");
                }

                modules.Add(module);
            }
        }

        var duplicates = modules
            .GroupBy(x => x.Name, StringComparer.OrdinalIgnoreCase)
            .Where(group => group.Count() > 1)
            .Select(group => group.Key)
            .ToArray();

        if (duplicates.Length > 0)
        {
            throw new ModuleLoadingException($"Duplicate module names found: {string.Join(", ", duplicates)}");
        }

        return modules;
    }
}
