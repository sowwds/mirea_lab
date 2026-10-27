using UnityEngine;

public sealed class CeilingFan : RoomAction
{
    public Transform rotor;
    public Transform airflowOrigin;
    public float airRadius = 2.2f;
    public float airHeight = 3f;
    public float windAcceleration = 16f;
    public Vector3 windDirection = new Vector3(1f, 0.08f, 0.5f);
    public bool IsRunning { get; private set; }
    public float CurrentSpeed { get; private set; }
    public override string Hint => IsRunning ? "Остановить вентилятор — E" : "Запустить вентилятор — E";
    private RoomMovable[] objects;
    private void Start() { objects = FindObjectsByType<RoomMovable>(FindObjectsSortMode.None); }
    public override void Use() { IsRunning = !IsRunning; }
    private void Update()
    {
        CurrentSpeed = Mathf.MoveTowards(CurrentSpeed, IsRunning ? 300f : 0f, 180f * Time.deltaTime);
        if (rotor != null) rotor.Rotate(Vector3.up, CurrentSpeed * Time.deltaTime, Space.Self);
    }
    private void FixedUpdate()
    {
        if (CurrentSpeed < 20f || objects == null || airflowOrigin == null) return;
        foreach (var item in objects)
        {
            if (item == null || !item.canLift) continue;
            var body = item.GetComponent<Rigidbody>();
            if (body == null || body.isKinematic) continue;
            Vector3 offset = body.worldCenterOfMass - airflowOrigin.position;
            float radius = new Vector2(offset.x, offset.z).magnitude;
            if (offset.y > 0f || offset.y < -airHeight || radius > airRadius) continue;
            float strength = Mathf.Clamp01(CurrentSpeed / 300f) * (1f - 0.45f * radius / airRadius);
            body.AddForce(windDirection.normalized * windAcceleration * strength, ForceMode.Acceleration);
            body.AddTorque(new Vector3(0.12f, 0.06f, -0.12f) * strength, ForceMode.Acceleration);
        }
    }
}
