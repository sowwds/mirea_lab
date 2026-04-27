using System.Diagnostics;
using proj4.Contracts;
using proj4.Health;
using proj4.Metrics;
using proj4.Models;

namespace proj4.Services;

public sealed class BookingProcessService(
    ILogger<BookingProcessService> logger,
    BookingMetrics metrics,
    BookingHealthState healthState) : IBookingProcessService
{
    private readonly object _gate = new();
    private readonly Dictionary<string, BookingProcess> _processes = new(StringComparer.Ordinal);

    public BookingEventResponse HandleEvent(BookingEventRequest request, string correlationId)
    {
        var stopwatch = Stopwatch.StartNew();

        lock (_gate)
        {
            var process = GetOrCreateProcess(request.ProcessKey);

            if (!process.ProcessedIdempotencyKeys.Add(request.IdempotencyKey))
            {
                metrics.RecordDuplicate();
                logger.LogInformation(
                    "Duplicate delivery detected for process {ProcessKey}, idempotency key {IdempotencyKey}, correlation {CorrelationId}",
                    request.ProcessKey,
                    request.IdempotencyKey,
                    correlationId);

                return new BookingEventResponse(
                    request.ProcessKey,
                    process.State,
                    process.State,
                    IsDuplicate: true,
                    CompensationApplied: false,
                    Message: "Duplicate delivery ignored.",
                    CorrelationId: correlationId);
            }

            var previousState = process.State;

            try
            {
                var result = ApplyTransition(process, request, correlationId);
                stopwatch.Stop();
                if (result.CompensationApplied)
                {
                    metrics.RecordFailure(request.EventType.ToString(), stopwatch.Elapsed.TotalMilliseconds);
                    healthState.RegisterCriticalFailure();
                }
                else
                {
                    metrics.RecordSuccess(request.EventType.ToString(), stopwatch.Elapsed.TotalMilliseconds);
                }

                return result with { PreviousState = previousState, CorrelationId = correlationId };
            }
            catch (BookingProcessException exception)
            {
                stopwatch.Stop();
                metrics.RecordFailure(request.EventType.ToString(), stopwatch.Elapsed.TotalMilliseconds);
                healthState.RegisterCriticalFailure();

                logger.LogError(
                    exception,
                    "Failed to process event {EventType} for process {ProcessKey}. CorrelationId: {CorrelationId}",
                    request.EventType,
                    request.ProcessKey,
                    correlationId);

                throw;
            }
        }
    }

    public BookingProcessSnapshot? GetSnapshot(string processKey)
    {
        lock (_gate)
        {
            if (!_processes.TryGetValue(processKey, out var process))
            {
                return null;
            }

            return new BookingProcessSnapshot(process.ProcessKey, process.State, process.ProcessedIdempotencyKeys.ToArray());
        }
    }

    private BookingEventResponse ApplyTransition(BookingProcess process, BookingEventRequest request, string correlationId)
    {
        return (process.State, request.EventType) switch
        {
            (BookingState.NotStarted, BookingEventType.StartBooking) => TransitionTo(
                process,
                BookingState.Requested,
                request,
                correlationId,
                "Booking process started."),

            (BookingState.Requested, BookingEventType.ReserveRoom) => TransitionTo(
                process,
                BookingState.RoomReserved,
                request,
                correlationId,
                "Room reserved."),

            (BookingState.RoomReserved, BookingEventType.SendNotification) when request.SimulateFailure => CompensateReservation(
                process,
                request,
                correlationId),

            (BookingState.RoomReserved, BookingEventType.SendNotification) => TransitionTo(
                process,
                BookingState.NotificationSent,
                request,
                correlationId,
                "Notification sent."),

            (BookingState.NotificationSent, BookingEventType.CompleteBooking) => TransitionTo(
                process,
                BookingState.Completed,
                request,
                correlationId,
                "Booking completed."),

            _ => throw new BookingProcessException(
                $"Invalid transition. Current state '{process.State}', event '{request.EventType}'.")
        };
    }

    private BookingEventResponse TransitionTo(
        BookingProcess process,
        BookingState nextState,
        BookingEventRequest request,
        string correlationId,
        string message)
    {
        var previousState = process.State;
        process.State = nextState;

        logger.LogInformation(
            "Transition {PreviousState} -> {NextState} for process {ProcessKey}. Event: {EventType}. CorrelationId: {CorrelationId}",
            previousState,
            nextState,
            request.ProcessKey,
            request.EventType,
            correlationId);

        return new BookingEventResponse(
            request.ProcessKey,
            previousState,
            nextState,
            IsDuplicate: false,
            CompensationApplied: false,
            Message: message,
            CorrelationId: correlationId);
    }

    private BookingEventResponse CompensateReservation(
        BookingProcess process,
        BookingEventRequest request,
        string correlationId)
    {
        var previousState = process.State;
        process.State = BookingState.Compensated;
        metrics.RecordCompensation();

        logger.LogWarning(
            "Compensation applied for process {ProcessKey}. State {PreviousState} -> {NextState}. Event: {EventType}. CorrelationId: {CorrelationId}",
            request.ProcessKey,
            previousState,
            process.State,
            request.EventType,
            correlationId);

        return new BookingEventResponse(
            request.ProcessKey,
            previousState,
            process.State,
            IsDuplicate: false,
            CompensationApplied: true,
            Message: "Notification step failed. Reservation compensated.",
            CorrelationId: correlationId);
    }

    private BookingProcess GetOrCreateProcess(string processKey)
    {
        if (_processes.TryGetValue(processKey, out var existing))
        {
            return existing;
        }

        var process = new BookingProcess
        {
            ProcessKey = processKey
        };

        _processes[processKey] = process;
        return process;
    }
}
