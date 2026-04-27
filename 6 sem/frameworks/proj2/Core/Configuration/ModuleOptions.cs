namespace proj2.Core.Configuration;

public sealed class ModuleOptions
{
    public const string SectionName = "Modules";

    public string ModulesPath { get; init; } = "modules";
    public List<string> EnabledModules { get; init; } = [];
}
