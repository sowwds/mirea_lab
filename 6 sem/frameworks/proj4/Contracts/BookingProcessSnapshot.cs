namespace proj4.Contracts;

public sealed record BookingProcessSnapshot(
    string ProcessKey,
    BookingState State,
    IReadOnlyCollection<string> ProcessedIdempotencyKeys);
