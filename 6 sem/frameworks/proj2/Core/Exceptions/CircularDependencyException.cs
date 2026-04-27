namespace proj2.Core.Exceptions;

public sealed class CircularDependencyException(string message) : Exception(message);
