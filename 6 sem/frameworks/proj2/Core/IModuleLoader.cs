using ModuleContract;

namespace proj2.Core;

public interface IModuleLoader
{
    IReadOnlyCollection<IAppModule> LoadModules(string modulesPath);
}
