using proj1.Contracts;

namespace proj1.Services;

public interface IStudyTaskService
{
    IReadOnlyCollection<StudyTaskDto> GetAll();
    StudyTaskDto? GetById(int id);
    StudyTaskDto Create(CreateStudyTaskRequest request);
}
