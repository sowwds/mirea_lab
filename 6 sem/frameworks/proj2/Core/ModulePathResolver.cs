namespace proj2.Core;

public static class ModulePathResolver
{
    public static string Resolve(string contentRootPath, string configuredPath)
    {
        return Path.GetFullPath(Path.Combine(contentRootPath, configuredPath));
    }
}
