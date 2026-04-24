namespace proj1.Exceptions;

public sealed class NotFoundException(string message) : Exception(message);
