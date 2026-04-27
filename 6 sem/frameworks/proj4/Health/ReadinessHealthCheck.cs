using Microsoft.Extensions.Diagnostics.HealthChecks;

namespace proj4.Health;

public sealed class ReadinessHealthCheck(BookingHealthState state) : IHealthCheck
{
    public Task<HealthCheckResult> CheckHealthAsync(HealthCheckContext context, CancellationToken cancellationToken = default)
    {
        return Task.FromResult(
            state.IsReady
                ? HealthCheckResult.Healthy("Service is ready.")
                : HealthCheckResult.Unhealthy("Service is critically degraded."));
    }
}
