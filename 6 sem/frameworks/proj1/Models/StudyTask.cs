namespace proj1.Models;

public sealed record StudyTask(
    int Id,
    string Title,
    string Subject,
    int Difficulty,
    int EstimatedHours);
