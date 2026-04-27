namespace proj2.Core;

public static class CommandLineConfigResolver
{
    public static string ResolveConfigFile(string[] args)
    {
        for (var i = 0; i < args.Length - 1; i++)
        {
            if (args[i] == "--config")
            {
                return args[i + 1];
            }
        }

        return "appsettings.json";
    }
}
