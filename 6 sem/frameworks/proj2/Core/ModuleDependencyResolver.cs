using ModuleContract;
using proj2.Core.Exceptions;

namespace proj2.Core;

public sealed class ModuleDependencyResolver : IModuleDependencyResolver
{
    public IReadOnlyList<IAppModule> Resolve(IEnumerable<string> enabledModules, IReadOnlyCollection<IAppModule> availableModules)
    {
        var requestedNames = enabledModules.Distinct(StringComparer.OrdinalIgnoreCase).ToArray();
        var availableByName = availableModules.ToDictionary(x => x.Name, StringComparer.OrdinalIgnoreCase);
        var ordered = new List<IAppModule>();
        var state = new Dictionary<string, VisitState>(StringComparer.OrdinalIgnoreCase);
        var stack = new Stack<string>();

        foreach (var moduleName in requestedNames)
        {
            Visit(moduleName);
        }

        return ordered;

        void Visit(string moduleName)
        {
            if (!availableByName.TryGetValue(moduleName, out var module))
            {
                throw new MissingModuleException($"Module '{moduleName}' is required but was not found.");
            }

            if (state.TryGetValue(moduleName, out var currentState))
            {
                if (currentState == VisitState.Visited)
                {
                    return;
                }

                if (currentState == VisitState.Visiting)
                {
                    var cycleItems = stack.Reverse().SkipWhile(x => !x.Equals(moduleName, StringComparison.OrdinalIgnoreCase)).Append(moduleName);
                    throw new CircularDependencyException($"Circular dependency detected: {string.Join(" -> ", cycleItems)}");
                }
            }

            state[moduleName] = VisitState.Visiting;
            stack.Push(moduleName);

            foreach (var dependency in module.Dependencies)
            {
                Visit(dependency);
            }

            stack.Pop();
            state[moduleName] = VisitState.Visited;
            ordered.Add(module);
        }
    }

    private enum VisitState
    {
        Visiting,
        Visited
    }
}
