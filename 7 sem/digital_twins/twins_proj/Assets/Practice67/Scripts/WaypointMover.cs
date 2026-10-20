using UnityEngine;

[RequireComponent(typeof(Rigidbody), typeof(Collider))]
public class WaypointMover : MonoBehaviour
{
    private static readonly int WalkingParameter = Animator.StringToHash("Walking");

    public Vector3[] waypoints;
    public float speed = 1.6f;
    public float turnSpeed = 240f;
    public float waitAtWaypoint = 0.8f;
    public float arrivalDistance = 0.25f;

    private Rigidbody body;
    private int targetIndex;
    private float waitTimer;
    private Transform leftLeg;
    private Transform rightLeg;
    private Animator animator;
    private float walkCycle;
    private bool isWalking;

    private void Awake()
    {
        body = GetComponent<Rigidbody>();
        body.isKinematic = true;
        body.useGravity = false;
        leftLeg = transform.Find("LeftLeg");
        rightLeg = transform.Find("RightLeg");
        animator = GetComponentInChildren<Animator>(true);
        if (animator != null)
        {
            animator.applyRootMotion = false;
            animator.SetBool(WalkingParameter, false);
        }
    }

    private void FixedUpdate()
    {
        if (waypoints == null || waypoints.Length < 2)
        {
            SetWalking(false);
            return;
        }

        Vector3 target = waypoints[targetIndex];
        Vector3 offset = target - body.position;
        offset.y = 0f;

        if (offset.magnitude <= arrivalDistance)
        {
            SetWalking(false);
            waitTimer += Time.fixedDeltaTime;
            if (waitTimer >= waitAtWaypoint)
            {
                waitTimer = 0f;
                targetIndex = (targetIndex + 1) % waypoints.Length;
            }
            return;
        }

        SetWalking(true);
        Vector3 direction = offset.normalized;
        Quaternion targetRotation = Quaternion.LookRotation(direction, Vector3.up);
        body.MoveRotation(Quaternion.RotateTowards(body.rotation, targetRotation, turnSpeed * Time.fixedDeltaTime));
        body.MovePosition(body.position + direction * speed * Time.fixedDeltaTime);

        if (animator == null)
        {
            walkCycle += Time.fixedDeltaTime * speed * 5f;
            float legAngle = Mathf.Sin(walkCycle) * 28f;
            if (leftLeg != null) leftLeg.localRotation = Quaternion.Euler(legAngle, 0f, 0f);
            if (rightLeg != null) rightLeg.localRotation = Quaternion.Euler(-legAngle, 0f, 0f);
        }
    }

    private void SetWalking(bool value)
    {
        if (isWalking == value)
        {
            return;
        }

        isWalking = value;
        if (animator != null)
        {
            animator.SetBool(WalkingParameter, value);
        }
    }
}
