using ModuleContract;

namespace proj2.Core;

public interface IModuleDependencyResolver
{
    IReadOnlyList<IAppModule> Resolve(IEnumerable<string> enabledModules, IReadOnlyCollection<IAppModule> availableModules);
}
