namespace proj4.Contracts;

public sealed record BookingEventRequest(
    string ProcessKey,
    string IdempotencyKey,
    BookingEventType EventType,
    bool SimulateFailure = false);
