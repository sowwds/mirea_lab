using System.Diagnostics.Metrics;

namespace proj4.Metrics;

public sealed class BookingMetrics
{
    private readonly Meter _meter = new("proj4.booking");
    private readonly Counter<long> _successfulTransitions;
    private readonly Counter<long> _failedTransitions;
    private readonly Counter<long> _duplicateDeliveries;
    private readonly Counter<long> _compensations;
    private readonly Histogram<double> _stepLatencyMs;

    private long _successfulTransitionsValue;
    private long _failedTransitionsValue;
    private long _duplicateDeliveriesValue;
    private long _compensationsValue;

    public BookingMetrics()
    {
        _successfulTransitions = _meter.CreateCounter<long>("booking_successful_transitions");
        _failedTransitions = _meter.CreateCounter<long>("booking_failed_transitions");
        _duplicateDeliveries = _meter.CreateCounter<long>("booking_duplicate_deliveries");
        _compensations = _meter.CreateCounter<long>("booking_compensations");
        _stepLatencyMs = _meter.CreateHistogram<double>("booking_step_latency_ms");
    }

    public void RecordSuccess(string step, double latencyMs)
    {
        _successfulTransitions.Add(1, new KeyValuePair<string, object?>("step", step));
        _stepLatencyMs.Record(latencyMs, new KeyValuePair<string, object?>("step", step));
        Interlocked.Increment(ref _successfulTransitionsValue);
    }

    public void RecordFailure(string step, double latencyMs)
    {
        _failedTransitions.Add(1, new KeyValuePair<string, object?>("step", step));
        _stepLatencyMs.Record(latencyMs, new KeyValuePair<string, object?>("step", step));
        Interlocked.Increment(ref _failedTransitionsValue);
    }

    public void RecordDuplicate()
    {
        _duplicateDeliveries.Add(1);
        Interlocked.Increment(ref _duplicateDeliveriesValue);
    }

    public void RecordCompensation()
    {
        _compensations.Add(1);
        Interlocked.Increment(ref _compensationsValue);
    }

    public MetricsSnapshot GetSnapshot()
    {
        return new MetricsSnapshot(
            SuccessfulTransitions: Interlocked.Read(ref _successfulTransitionsValue),
            FailedTransitions: Interlocked.Read(ref _failedTransitionsValue),
            DuplicateDeliveries: Interlocked.Read(ref _duplicateDeliveriesValue),
            Compensations: Interlocked.Read(ref _compensationsValue));
    }
}
