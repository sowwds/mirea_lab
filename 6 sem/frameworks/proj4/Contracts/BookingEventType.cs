using System.Text.Json.Serialization;

namespace proj4.Contracts;

[JsonConverter(typeof(JsonStringEnumConverter))]
public enum BookingEventType
{
    StartBooking,
    ReserveRoom,
    SendNotification,
    CompleteBooking
}
