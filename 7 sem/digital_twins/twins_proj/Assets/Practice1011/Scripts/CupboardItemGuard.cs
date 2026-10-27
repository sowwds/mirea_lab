using UnityEngine;

public sealed class CupboardItemGuard : MonoBehaviour
{
    public HingedDoor door;
    public bool CanTake => door == null || door.IsAccessible;
}
