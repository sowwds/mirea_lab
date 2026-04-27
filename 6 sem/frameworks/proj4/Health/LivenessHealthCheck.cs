using Microsoft.Extensions.Diagnostics.HealthChecks;

namespace proj4.Health;

public sealed class LivenessHealthCheck : IHealthCheck
{
    public Task<HealthCheckResult> CheckHealthAsync(HealthCheckContext context, CancellationToken cancellationToken = default)
    {
        return Task.FromResult(HealthCheckResult.Healthy("Service is alive."));
    }
}
