public sealed class FanControl : RoomAction
{
    public CeilingFan fan;
    public override string Hint => fan != null ? fan.Hint : "";
    public override void Use() { if (fan != null) fan.Use(); }
}
