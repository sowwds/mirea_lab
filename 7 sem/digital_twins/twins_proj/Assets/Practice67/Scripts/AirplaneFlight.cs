using UnityEngine;

[RequireComponent(typeof(Rigidbody), typeof(Collider))]
public class AirplaneFlight : MonoBehaviour
{
    public Vector3[] waypoints;
    public float cruiseSpeed = 22f;
    public float turnSpeed = 38f;
    public float arrivalDistance = 8f;
    public float bankAngle = 18f;

    private Rigidbody body;
    private int targetIndex;

    private void Awake()
    {
        body = GetComponent<Rigidbody>();
        body.isKinematic = true;
        body.useGravity = false;
    }

    private void FixedUpdate()
    {
        if (waypoints == null || waypoints.Length < 3) return;

        Vector3 target = waypoints[targetIndex];
        Vector3 direction = target - body.position;
        if (direction.magnitude <= arrivalDistance)
        {
            targetIndex = (targetIndex + 1) % waypoints.Length;
            target = waypoints[targetIndex];
            direction = target - body.position;
        }

        direction.Normalize();
        Quaternion lookRotation = Quaternion.LookRotation(direction, Vector3.up);
        float turnDirection = Vector3.SignedAngle(transform.forward, direction, Vector3.up);
        float bank = Mathf.Clamp(-turnDirection, -bankAngle, bankAngle);
        Quaternion bankedRotation = lookRotation * Quaternion.Euler(0f, 0f, bank);
        body.MoveRotation(Quaternion.RotateTowards(body.rotation, bankedRotation, turnSpeed * Time.fixedDeltaTime));

        float verticalFactor = Mathf.Clamp01(Mathf.Abs(direction.y) * 2f);
        float currentSpeed = Mathf.Lerp(cruiseSpeed, cruiseSpeed * 0.78f, verticalFactor);
        body.MovePosition(body.position + transform.forward * currentSpeed * Time.fixedDeltaTime);
    }
}
