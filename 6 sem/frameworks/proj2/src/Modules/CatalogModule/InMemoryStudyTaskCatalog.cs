using ModuleContract;

namespace CatalogModule;

internal sealed class InMemoryStudyTaskCatalog : IStudyTaskCatalog
{
    private static readonly IReadOnlyCollection<StudyTaskItem> Items =
    [
        new("Build middleware prototype", "Frameworks", 5),
        new("Write dependency tests", "Testing", 4),
        new("Prepare module demo", "Architecture", 6)
    ];

    public IReadOnlyCollection<StudyTaskItem> GetAll() => Items;
}
