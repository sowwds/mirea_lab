using proj1.Contracts;
using proj1.Exceptions;
using proj1.Models;

namespace proj1.Services;

public sealed class InMemoryStudyTaskService : IStudyTaskService
{
    private readonly object _gate = new();
    private readonly List<StudyTask> _items =
    [
        new(1, "Build middleware chain", "Frameworks", 5, 4),
        new(2, "Write validation tests", "Testing", 4, 3)
    ];

    private int _nextId = 3;

    public IReadOnlyCollection<StudyTaskDto> GetAll()
    {
        lock (_gate)
        {
            return _items
                .Select(Map)
                .ToArray();
        }
    }

    public StudyTaskDto? GetById(int id)
    {
        lock (_gate)
        {
            var item = _items.FirstOrDefault(x => x.Id == id);
            return item is null ? null : Map(item);
        }
    }

    public StudyTaskDto Create(CreateStudyTaskRequest request)
    {
        Validate(request);

        lock (_gate)
        {
            var item = new StudyTask(
                _nextId++,
                request.Title.Trim(),
                request.Subject.Trim(),
                request.Difficulty,
                request.EstimatedHours);

            _items.Add(item);
            return Map(item);
        }
    }

    private static void Validate(CreateStudyTaskRequest request)
    {
        if (string.IsNullOrWhiteSpace(request.Title))
        {
            throw new DomainValidationException("Title must not be empty.");
        }

        if (string.IsNullOrWhiteSpace(request.Subject))
        {
            throw new DomainValidationException("Subject must not be empty.");
        }

        if (request.Difficulty is < 1 or > 10)
        {
            throw new DomainValidationException("Difficulty must be between 1 and 10.");
        }

        if (request.EstimatedHours < 0)
        {
            throw new DomainValidationException("Estimated hours must not be negative.");
        }
    }

    private static StudyTaskDto Map(StudyTask item)
    {
        return new StudyTaskDto(
            item.Id,
            item.Title,
            item.Subject,
            item.Difficulty,
            item.EstimatedHours);
    }
}
