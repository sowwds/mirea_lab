using UnityEngine;

public sealed class WindowShutters : RoomAction
{
    public Transform[] slats;
    public Transform crank;
    public Light daylight;
    public Transform leftShutter;
    public Transform rightShutter;
    public bool IsOpen { get; private set; }
    public override string Hint => IsOpen ? "Закрыть ставни — E" : "Открыть ставни — E";
    private float angle;
    private void Start() { if (daylight != null) daylight.intensity = 0f; }
    public override void Use() { IsOpen = !IsOpen; }
    private void Update()
    {
        angle = Mathf.MoveTowards(angle, IsOpen ? 78f : 0f, 85f * Time.deltaTime);
        foreach (var slat in slats) if (slat != null) slat.localRotation = Quaternion.Euler(angle, 0f, 0f);
        if (leftShutter != null) leftShutter.localRotation = Quaternion.Euler(0f, angle * 110f / 78f, 0f);
        if (rightShutter != null) rightShutter.localRotation = Quaternion.Euler(0f, -angle * 110f / 78f, 0f);
        if (crank != null) crank.localRotation = Quaternion.Euler(0f, 0f, angle * 3f);
        if (daylight != null) daylight.intensity = Mathf.Lerp(0f, 1.6f, angle / 78f);
    }
}
