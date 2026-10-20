using System.IO;
using UnityEditor;

namespace Hodaart.LowPolyCharacterCollection3.Editor
{
    /// <summary>
    /// Checks whether the Input System package is installed and informs the user
    /// if it is required by this asset.
    /// </summary>
    [InitializeOnLoad]
    internal static class HodaartDependencyInstaller
    {
        private const string PackageId = "com.unity.inputsystem";
        private const string ActiveInputHandlerKey = "activeInputHandler:";
        private const string EditorPrefKey =
            "Hodaart.LowPolyCharacterCollection3.DependencyChecked";

        static HodaartDependencyInstaller()
        {
            // Only run once per project
            if (EditorPrefs.GetBool(EditorPrefKey, false))
                return;

            EditorPrefs.SetBool(EditorPrefKey, true);

            if (!IsInManifest(PackageId))
            {
                ShowInstallMessage();
                return;
            }

            PromptInputHandlingIfNeeded();
        }

        private static bool IsInManifest(string packageId)
        {
            string path = Path.GetFullPath("Packages/manifest.json");

            if (!File.Exists(path))
                return false;

            return File.ReadAllText(path).Contains($"\"{packageId}\"");
        }

        private static void ShowInstallMessage()
        {
            bool open = EditorUtility.DisplayDialog(
                "Input System Required",
                "This asset requires Unity's Input System package.\n\n" +
                "Please install it using:\n\n" +
                "Window → Package Manager\n" +
                "→ Unity Registry\n" +
                "→ Input System\n\n" +
                "After installation, restart Unity if requested.",
                "Open Package Manager",
                "Later");

            if (open)
                EditorApplication.ExecuteMenuItem("Window/Package Manager");
        }

        private static void PromptInputHandlingIfNeeded()
        {
            if (!IsUsingOldInputManagerOnly())
                return;

            bool open = EditorUtility.DisplayDialog(
                "Input System Detected",
                "The Input System package is installed.\n\n" +
                "Your project is still using the old Input Manager.\n\n" +
                "For this asset to work correctly, set Active Input Handling to:\n\n" +
                "• Both\nor\n• Input System Package (New)",
                "Open Project Settings",
                "Later");

            if (open)
                SettingsService.OpenProjectSettings("Project/Player");
        }

        private static bool IsUsingOldInputManagerOnly()
        {
            string path = Path.GetFullPath("ProjectSettings/ProjectSettings.asset");

            if (!File.Exists(path))
                return false;

            foreach (string line in File.ReadAllLines(path))
            {
                string trimmed = line.Trim();

                if (!trimmed.StartsWith(ActiveInputHandlerKey))
                    continue;

                string value = trimmed.Substring(ActiveInputHandlerKey.Length).Trim();

                return value == "0";
            }

            return false;
        }
    }
}