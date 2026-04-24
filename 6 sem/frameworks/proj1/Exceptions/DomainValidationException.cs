namespace proj1.Exceptions;

public sealed class DomainValidationException(string message) : Exception(message);
