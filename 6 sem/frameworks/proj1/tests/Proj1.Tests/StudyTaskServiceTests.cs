using proj1.Contracts;
using proj1.Exceptions;
using proj1.Services;

namespace Proj1.Tests;

public sealed class StudyTaskServiceTests
{
    private readonly InMemoryStudyTaskService _service = new();

    [Fact]
    public void Create_ShouldThrow_WhenTitleIsBlank()
    {
        var request = new CreateStudyTaskRequest("", "Frameworks", 4, 2);

        var action = () => _service.Create(request);

        var exception = Assert.Throws<DomainValidationException>(action);
        Assert.Equal("Title must not be empty.", exception.Message);
    }

    [Fact]
    public void Create_ShouldThrow_WhenDifficultyIsOutOfRange()
    {
        var request = new CreateStudyTaskRequest("Task", "Frameworks", 11, 2);

        var action = () => _service.Create(request);

        var exception = Assert.Throws<DomainValidationException>(action);
        Assert.Equal("Difficulty must be between 1 and 10.", exception.Message);
    }

    [Fact]
    public void Create_ShouldTrimValues_AndStoreTask()
    {
        var created = _service.Create(new CreateStudyTaskRequest("  Task  ", "  Testing  ", 3, 2));

        var fetched = _service.GetById(created.Id);

        Assert.NotNull(fetched);
        Assert.Equal("Task", fetched!.Title);
        Assert.Equal("Testing", fetched.Subject);
    }
}
