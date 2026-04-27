namespace proj4.Metrics;

public sealed record MetricsSnapshot(
    long SuccessfulTransitions,
    long FailedTransitions,
    long DuplicateDeliveries,
    long Compensations);
