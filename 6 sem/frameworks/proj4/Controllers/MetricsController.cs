using Microsoft.AspNetCore.Mvc;
using proj4.Metrics;

namespace proj4.Controllers;

[ApiController]
[Route("metrics-summary")]
public sealed class MetricsController(BookingMetrics metrics) : ControllerBase
{
    [HttpGet]
    public ActionResult<MetricsSnapshot> Get()
    {
        return Ok(metrics.GetSnapshot());
    }
}
