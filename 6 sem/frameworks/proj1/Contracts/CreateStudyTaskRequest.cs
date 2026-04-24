namespace proj1.Contracts;

public sealed record CreateStudyTaskRequest(
    string Title,
    string Subject,
    int Difficulty,
    int EstimatedHours);
