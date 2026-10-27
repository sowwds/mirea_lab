using UnityEngine;

[RequireComponent(typeof(BoxCollider))]
public sealed class CityRoomPortal : MonoBehaviour
{
    public Transform destination;
    public HingedDoor requiredOpenDoor;
    public int TransferCount { get; private set; }
    private void OnTriggerEnter(Collider other)
    {
        var player = other.GetComponentInParent<CityWalker>();
        if (player != null) Transfer(player);
    }
    private void OnTriggerStay(Collider other)
    {
        var player = other.GetComponentInParent<CityWalker>();
        if (player != null) Transfer(player);
    }
    public void Transfer(CityWalker player)
    {
        if (destination == null || Time.time < player.PortalReadyAt) return;
        if (requiredOpenDoor != null && !requiredOpenDoor.IsAccessible) return;
        player.Teleport(destination.position, destination.rotation);
        TransferCount++;
    }
}
