using proj3.Contracts;

namespace proj3.Services;

public sealed class InMemoryStudyItemService : IStudyItemService
{
    private static readonly IReadOnlyCollection<StudyItemDto> Items =
    [
        new(1, "Prepare architecture notes", "Frameworks"),
        new(2, "Check configuration priority", "Deployment")
    ];

    public IReadOnlyCollection<StudyItemDto> GetAll() => Items;
}
