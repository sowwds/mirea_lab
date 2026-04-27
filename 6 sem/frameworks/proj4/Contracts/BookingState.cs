using System.Text.Json.Serialization;

namespace proj4.Contracts;

[JsonConverter(typeof(JsonStringEnumConverter))]
public enum BookingState
{
    NotStarted,
    Requested,
    RoomReserved,
    NotificationSent,
    Completed,
    Compensated
}
