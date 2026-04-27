namespace proj4.Contracts;

public sealed record BookingEventResponse(
    string ProcessKey,
    BookingState PreviousState,
    BookingState CurrentState,
    bool IsDuplicate,
    bool CompensationApplied,
    string Message,
    string CorrelationId);
