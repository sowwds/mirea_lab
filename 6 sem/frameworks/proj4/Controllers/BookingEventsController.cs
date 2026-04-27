using Microsoft.AspNetCore.Mvc;
using proj4.Contracts;
using proj4.Middlewares;
using proj4.Services;

namespace proj4.Controllers;

[ApiController]
[Route("api/bookings")]
public sealed class BookingEventsController(IBookingProcessService bookingProcessService) : ControllerBase
{
    [HttpPost("events")]
    public ActionResult<BookingEventResponse> HandleEvent([FromBody] BookingEventRequest request)
    {
        var correlationId = HttpContext.Items[CorrelationIdMiddleware.CorrelationIdKey]?.ToString() ?? HttpContext.TraceIdentifier;
        try
        {
            var result = bookingProcessService.HandleEvent(request, correlationId);
            return Ok(result);
        }
        catch (BookingProcessException exception)
        {
            return BadRequest(new
            {
                code = "invalid_transition",
                message = exception.Message,
                correlationId
            });
        }
    }

    [HttpGet("{processKey}")]
    public ActionResult<BookingProcessSnapshot> GetProcess(string processKey)
    {
        var snapshot = bookingProcessService.GetSnapshot(processKey);
        return snapshot is null ? NotFound() : Ok(snapshot);
    }
}
