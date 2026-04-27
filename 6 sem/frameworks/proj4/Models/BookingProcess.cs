using proj4.Contracts;

namespace proj4.Models;

public sealed class BookingProcess
{
    public string ProcessKey { get; init; } = string.Empty;
    public BookingState State { get; set; } = BookingState.NotStarted;
    public HashSet<string> ProcessedIdempotencyKeys { get; } = new(StringComparer.Ordinal);
}
