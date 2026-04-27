using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using proj4.Configuration;
using proj4.Contracts;
using proj4.Health;
using proj4.Metrics;
using proj4.Services;

namespace Proj4.Tests;

public sealed class BookingProcessServiceTests
{
    private readonly BookingMetrics _metrics = new();
    private readonly BookingHealthState _healthState = new(Options.Create(new OperationsOptions
    {
        CriticalFailureThreshold = 2
    }));

    [Fact]
    public void HandleEvent_ShouldProcessHappyPath()
    {
        var service = CreateService();

        service.HandleEvent(new BookingEventRequest("p1", "e1", BookingEventType.StartBooking), "corr-1");
        service.HandleEvent(new BookingEventRequest("p1", "e2", BookingEventType.ReserveRoom), "corr-2");
        service.HandleEvent(new BookingEventRequest("p1", "e3", BookingEventType.SendNotification), "corr-3");
        var completion = service.HandleEvent(new BookingEventRequest("p1", "e4", BookingEventType.CompleteBooking), "corr-4");

        Assert.Equal(BookingState.Completed, completion.CurrentState);
        Assert.False(completion.IsDuplicate);
        Assert.False(completion.CompensationApplied);
    }

    [Fact]
    public void HandleEvent_ShouldIgnoreDuplicateDelivery()
    {
        var service = CreateService();

        service.HandleEvent(new BookingEventRequest("p1", "e1", BookingEventType.StartBooking), "corr-1");
        var duplicate = service.HandleEvent(new BookingEventRequest("p1", "e1", BookingEventType.StartBooking), "corr-2");

        Assert.True(duplicate.IsDuplicate);
        Assert.Equal(BookingState.Requested, duplicate.CurrentState);
        Assert.Equal(1, _metrics.GetSnapshot().DuplicateDeliveries);
    }

    [Fact]
    public void HandleEvent_ShouldCompensate_WhenNotificationStepFails()
    {
        var service = CreateService();

        service.HandleEvent(new BookingEventRequest("p1", "e1", BookingEventType.StartBooking), "corr-1");
        service.HandleEvent(new BookingEventRequest("p1", "e2", BookingEventType.ReserveRoom), "corr-2");
        var compensated = service.HandleEvent(new BookingEventRequest("p1", "e3", BookingEventType.SendNotification, SimulateFailure: true), "corr-3");

        Assert.True(compensated.CompensationApplied);
        Assert.Equal(BookingState.Compensated, compensated.CurrentState);

        var snapshot = _metrics.GetSnapshot();
        Assert.Equal(1, snapshot.Compensations);
        Assert.Equal(1, snapshot.FailedTransitions);
    }

    [Fact]
    public void HandleEvent_ShouldThrow_WhenTransitionIsInvalid()
    {
        var service = CreateService();

        var action = () => service.HandleEvent(new BookingEventRequest("p1", "e1", BookingEventType.CompleteBooking), "corr-1");

        var exception = Assert.Throws<BookingProcessException>(action);
        Assert.Contains("Invalid transition", exception.Message);
    }

    private BookingProcessService CreateService()
    {
        return new BookingProcessService(
            new NullLogger<BookingProcessService>(),
            _metrics,
            _healthState);
    }
}
