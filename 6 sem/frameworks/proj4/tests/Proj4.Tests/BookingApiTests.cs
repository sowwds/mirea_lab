using System.Net;
using System.Net.Http.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using proj4.Contracts;
using proj4.Metrics;

namespace Proj4.Tests;

public sealed class BookingApiTests
{
    [Fact]
    public async Task EventEndpoint_ShouldReturnCorrelationId_AndPersistProcessState()
    {
        using var factory = new WebApplicationFactory<Program>();
        using var client = factory.CreateClient();

        var response = await client.PostAsJsonAsync("/api/bookings/events", new BookingEventRequest(
            "room-201",
            "evt-1",
            BookingEventType.StartBooking));

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.True(response.Headers.Contains("X-Correlation-Id"));

        var body = await response.Content.ReadFromJsonAsync<BookingEventResponse>();
        Assert.NotNull(body);
        Assert.False(string.IsNullOrWhiteSpace(body!.CorrelationId));

        var snapshot = await client.GetFromJsonAsync<BookingProcessSnapshot>("/api/bookings/room-201");
        Assert.NotNull(snapshot);
        Assert.Equal(BookingState.Requested, snapshot!.State);
    }

    [Fact]
    public async Task Readiness_ShouldBecomeUnhealthy_AfterCriticalDegradation()
    {
        using var factory = new WebApplicationFactory<Program>();
        using var client = factory.CreateClient();

        for (var i = 0; i < 2; i++)
        {
            var invalid = await client.PostAsJsonAsync("/api/bookings/events", new BookingEventRequest(
                "room-bad",
                $"evt-{i}",
                BookingEventType.CompleteBooking));

            Assert.Equal(HttpStatusCode.BadRequest, invalid.StatusCode);
        }

        var ready = await client.GetAsync("/health/ready");

        Assert.Equal(HttpStatusCode.ServiceUnavailable, ready.StatusCode);
    }

    [Fact]
    public async Task MetricsEndpoint_ShouldExposeDuplicateAndCompensationCounters()
    {
        using var factory = new WebApplicationFactory<Program>();
        using var client = factory.CreateClient();

        await client.PostAsJsonAsync("/api/bookings/events", new BookingEventRequest("p1", "e1", BookingEventType.StartBooking));
        await client.PostAsJsonAsync("/api/bookings/events", new BookingEventRequest("p1", "e2", BookingEventType.ReserveRoom));
        await client.PostAsJsonAsync("/api/bookings/events", new BookingEventRequest("p1", "e2", BookingEventType.ReserveRoom));
        await client.PostAsJsonAsync("/api/bookings/events", new BookingEventRequest("p1", "e3", BookingEventType.SendNotification, true));

        var metrics = await client.GetFromJsonAsync<MetricsSnapshot>("/metrics-summary");

        Assert.NotNull(metrics);
        Assert.Equal(1, metrics!.DuplicateDeliveries);
        Assert.Equal(1, metrics.Compensations);
        Assert.Equal(1, metrics.FailedTransitions);
    }
}
