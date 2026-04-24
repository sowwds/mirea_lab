namespace proj1.Contracts;

public sealed record StudyTaskDto(
    int Id,
    string Title,
    string Subject,
    int Difficulty,
    int EstimatedHours);
