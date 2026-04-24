namespace proj1.Contracts;

public sealed record ErrorResponse(
    string Code,
    string Message,
    string RequestId);
