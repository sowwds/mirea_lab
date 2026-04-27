using proj3.Contracts;

namespace proj3.Services;

public interface IStudyItemService
{
    IReadOnlyCollection<StudyItemDto> GetAll();
}
