using UnityEngine;

public sealed class HingedDoor : RoomAction
{
    public float openAngle = 100f;
    public float degreesPerSecond = 110f;
    public string displayName = "дверь";
    public bool IsOpen { get; private set; }
    public bool IsAccessible => Mathf.Abs(angle) > 65f;
    public override string Hint => (IsOpen ? "Закрыть " : "Открыть ") + displayName + " — E";
    private Quaternion closedRotation;
    private float angle;
    private Rigidbody body;

    private void Awake()
    {
        closedRotation = transform.localRotation;
        body = GetComponent<Rigidbody>();
    }
    public override void Use() { IsOpen = !IsOpen; }
    private void FixedUpdate()
    {
        float nextAngle = Mathf.MoveTowards(angle, IsOpen ? openAngle : 0f, degreesPerSecond * Time.fixedDeltaTime);
        Quaternion local = closedRotation * Quaternion.Euler(0f, nextAngle, 0f);
        Quaternion world = transform.parent != null ? transform.parent.rotation * local : local;
        // A closed door never sweeps through the player. Pause until the doorway is clear.
        if (body != null && Mathf.Abs(nextAngle) < Mathf.Abs(angle))
        {
            var player = FindFirstObjectByType<CityWalker>();
            if (player != null && Vector3.Distance(player.transform.position, transform.position) < 1.25f) return;
        }
        angle = nextAngle;
        if (body != null) body.MoveRotation(world);
        else transform.localRotation = local;
    }
}
