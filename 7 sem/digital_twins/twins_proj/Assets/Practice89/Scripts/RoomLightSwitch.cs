using UnityEngine;

public sealed class RoomLightSwitch : MonoBehaviour
{
    public Light[] roomLights;
    public Transform lever;
    public bool IsOn { get; private set; } = true;
    public void Toggle()
    {
        IsOn = !IsOn;
        foreach (var light in roomLights) if (light != null) light.enabled = IsOn;
        if (lever != null) lever.localRotation = Quaternion.Euler(IsOn ? -25f : 25f, 0f, 0f);
    }
}
