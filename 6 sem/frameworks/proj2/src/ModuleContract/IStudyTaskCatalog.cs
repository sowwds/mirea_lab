namespace ModuleContract;

public interface IStudyTaskCatalog
{
    IReadOnlyCollection<StudyTaskItem> GetAll();
}
