using proj4.Contracts;

namespace proj4.Services;

public interface IBookingProcessService
{
    BookingEventResponse HandleEvent(BookingEventRequest request, string correlationId);
    BookingProcessSnapshot? GetSnapshot(string processKey);
}
