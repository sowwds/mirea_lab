using ModuleContract;

namespace ReportModule;

internal sealed class StudyTaskReportRenderer(IStudyTaskCatalog catalog) : IReportRenderer
{
    public string Render()
    {
        var tasks = catalog.GetAll()
            .OrderByDescending(x => x.Difficulty)
            .Select(x => $"- {x.Title} [{x.Subject}] difficulty={x.Difficulty}");

        return string.Join(Environment.NewLine, tasks);
    }
}
