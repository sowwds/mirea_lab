using Microsoft.Extensions.Options;
using proj4.Configuration;

namespace proj4.Health;

public sealed class BookingHealthState(IOptions<OperationsOptions> options)
{
    private int _criticalFailures;

    public bool IsReady => _criticalFailures < options.Value.CriticalFailureThreshold;

    public void RegisterCriticalFailure()
    {
        Interlocked.Increment(ref _criticalFailures);
    }
}
