using UnityEngine;
using UnityEngine.InputSystem;

public enum ConstructionMachineKind { Bulldozer, Excavator, Crane, Truck }

[RequireComponent(typeof(Rigidbody))]
public sealed class ConstructionMachine : MonoBehaviour
{
    public ConstructionMachineKind kind;
    public string displayName;
    public Transform upper, boom, stick, tool, bed, tailgate, hook, cable, cableTop;
    public Transform[] wheels;
    public Transform selectionRing;
    public float driveSpeed = 3f;
    public Vector3 footprint = new Vector3(2.5f, 2f, 4f);
    public bool IsSelected { get; private set; }
    public bool HasLoad => attachedLoad != null;
    private Rigidbody body;
    private Quaternion boomRest, stickRest, toolRest, bedRest, gateRest;
    private float upperAngle, boomAngle, stickAngle, toolAngle, bedAngle;
    private float ropeLength = 2.6f;
    private ConfigurableJoint loadJoint;
    private Rigidbody attachedLoad;
    private LineRenderer rope;
    private Vector3 hookVelocity;
    private Vector3 previousHook;

    public string Controls
    {
        get
        {
            switch (kind)
            {
                case ConstructionMachineKind.Excavator:
                    return "W / S — ход  •  A / D — поворот  •  Q / E — платформа\nR / F — стрела  •  T / G — рукоять  •  Y / H — ковш\nB — копать при опущенном ковше  •  Space — высыпать землю";
                case ConstructionMachineKind.Bulldozer:
                    return "W / S — ход  •  A / D — поворот  •  R / F — поднять / опустить отвал\nОпустите отвал и собирайте камни, двигаясь вперёд.";
                case ConstructionMachineKind.Truck:
                    return "W / S — ход  •  A / D — поворот  •  R / F — поднять / опустить кузов\nПри подъёме кузова задний борт открывается для разгрузки.";
                default:
                    return "W / S — ход  •  A / D — поворот  •  Q / E — поворот крана\nR / F — наклон стрелы  •  T / G — крюк вверх / вниз  •  Space — зацепить / отпустить груз";
            }
        }
    }

    private void Awake()
    {
        body = GetComponent<Rigidbody>();
        body.isKinematic = true;
        body.useGravity = false;
        if (boom != null) boomRest = boom.localRotation;
        if (stick != null) stickRest = stick.localRotation;
        if (tool != null) toolRest = tool.localRotation;
        if (bed != null) bedRest = bed.localRotation;
        if (tailgate != null) gateRest = tailgate.localRotation;
        if (hook != null) previousHook = hook.position;
        if (cable != null) rope = cable.GetComponent<LineRenderer>();
        SetSelected(false);
    }

    public void SetSelected(bool selected)
    {
        IsSelected = selected;
        if (selectionRing != null) selectionRing.gameObject.SetActive(selected);
    }

    private void Update()
    {
        if (IsSelected && Keyboard.current != null && kind == ConstructionMachineKind.Crane
            && Keyboard.current.spaceKey.wasPressedThisFrame) ToggleHook();
    }

    private void FixedUpdate()
    {
        if (IsSelected && Keyboard.current != null)
        {
            var k = Keyboard.current;
            Drive(Axis(k.wKey.isPressed, k.sKey.isPressed), Axis(k.dKey.isPressed, k.aKey.isPressed), Time.fixedDeltaTime);
            Operate(Axis(k.eKey.isPressed, k.qKey.isPressed), Axis(k.rKey.isPressed, k.fKey.isPressed),
                Axis(k.tKey.isPressed, k.gKey.isPressed), Axis(k.yKey.isPressed, k.hKey.isPressed), Time.fixedDeltaTime);
        }
        if (kind == ConstructionMachineKind.Crane) UpdateHook();
    }

    private static float Axis(bool positive, bool negative) => (positive ? 1f : 0f) - (negative ? 1f : 0f);

    public void Drive(float forward, float turn, float dt)
    {
        Quaternion nextRotation = body.rotation * Quaternion.Euler(0, turn * 40f * dt, 0);
        Vector3 delta = nextRotation * Vector3.forward * (forward * driveSpeed * dt);
        Vector3 next = body.position + delta;
        if (Mathf.Abs(next.x) > 23f || Mathf.Abs(next.z) > 18.5f) return;
        if (delta.sqrMagnitude > 0.000001f)
        {
            var hits = Physics.BoxCastAll(body.position + Vector3.up * 1.25f,
                new Vector3(footprint.x * .42f, .55f, footprint.z * .42f), delta.normalized,
                nextRotation, delta.magnitude + .06f, ~0, QueryTriggerInteraction.Ignore);
            foreach (var hit in hits)
            {
                if (hit.collider.transform.IsChildOf(transform)) continue;
                if (hit.collider.GetComponentInParent<LooseMaterial>() != null) continue;
                if (hit.collider.GetComponentInParent<LiftableCargo>() != null) continue;
                if (hit.collider.attachedRigidbody == null || hit.collider.attachedRigidbody.isKinematic) return;
            }
        }
        body.MoveRotation(nextRotation);
        body.MovePosition(next);
        if (wheels != null)
            foreach (var wheel in wheels) if (wheel != null) wheel.Rotate(Vector3.right, forward * driveSpeed * dt * 100f, Space.Self);
    }

    public void Operate(float slew, float lift, float extension, float curl, float dt)
    {
        if (upper != null)
        {
            upperAngle += slew * 35f * dt;
            upper.localRotation = Quaternion.Euler(0, upperAngle, 0);
        }
        if (kind == ConstructionMachineKind.Truck)
        {
            bedAngle = Mathf.Clamp(bedAngle - lift * 24f * dt, -52f, 0f);
            bed.localRotation = bedRest * Quaternion.Euler(bedAngle, 0, 0);
            if (tailgate != null) tailgate.localRotation = gateRest * Quaternion.Euler(Mathf.Abs(bedAngle) > 10f ? -80f : 0f, 0, 0);
        }
        else if (kind == ConstructionMachineKind.Excavator)
        {
            boomAngle = Mathf.Clamp(boomAngle - lift * 22f * dt, -30f, 28f);
            stickAngle = Mathf.Clamp(stickAngle - extension * 28f * dt, -65f, 65f);
            toolAngle = Mathf.Clamp(toolAngle + curl * 35f * dt, -85f, 85f);
            boom.localRotation = boomRest * Quaternion.Euler(boomAngle, 0, 0);
            stick.localRotation = stickRest * Quaternion.Euler(stickAngle, 0, 0);
            tool.localRotation = toolRest * Quaternion.Euler(toolAngle, 0, 0);
        }
        else if (kind == ConstructionMachineKind.Bulldozer)
        {
            boomAngle = Mathf.Clamp(boomAngle - lift * 22f * dt, -28f, 5f);
            boom.localRotation = boomRest * Quaternion.Euler(boomAngle, 0, 0);
        }
        else
        {
            boomAngle = Mathf.Clamp(boomAngle - lift * 18f * dt, -38f, 12f);
            boom.localRotation = boomRest * Quaternion.Euler(boomAngle, 0, 0);
            ropeLength = Mathf.Clamp(ropeLength - extension * 1.5f * dt, .6f, 6f);
        }
    }

    private void UpdateHook()
    {
        if (hook == null || cableTop == null) return;
        Vector3 target = cableTop.position + Vector3.down * ropeLength;
        target.y = Mathf.Max(.5f, target.y);
        var hookBody = hook.GetComponent<Rigidbody>();
        if (hookBody != null) hookBody.MovePosition(target);
        else hook.position = target;
        hookVelocity = (target - previousHook) / Time.fixedDeltaTime;
        previousHook = target;
        if (rope != null)
        {
            rope.SetPosition(0, cableTop.position);
            rope.SetPosition(1, target);
        }
    }

    public void ToggleHook()
    {
        if (loadJoint != null)
        {
            Destroy(loadJoint);
            loadJoint = null;
            attachedLoad = null;
            return;
        }
        if (hook == null) return;
        LiftableCargo closest = null;
        float distance = 1.5f;
        foreach (var collider in Physics.OverlapSphere(hook.position, 1.5f))
        {
            var cargo = collider.GetComponentInParent<LiftableCargo>();
            if (cargo == null) continue;
            float d = Vector3.Distance(hook.position, cargo.liftingPoint.position);
            if (d < distance) { closest = cargo; distance = d; }
        }
        if (closest == null) return;
        attachedLoad = closest.GetComponent<Rigidbody>();
        loadJoint = attachedLoad.gameObject.AddComponent<ConfigurableJoint>();
        loadJoint.connectedBody = hook.GetComponent<Rigidbody>();
        loadJoint.autoConfigureConnectedAnchor = false;
        loadJoint.anchor = attachedLoad.transform.InverseTransformPoint(closest.liftingPoint.position);
        loadJoint.connectedAnchor = Vector3.zero;
        loadJoint.xMotion = loadJoint.yMotion = loadJoint.zMotion = ConfigurableJointMotion.Locked;
        loadJoint.angularXMotion = loadJoint.angularYMotion = loadJoint.angularZMotion = ConfigurableJointMotion.Free;
        loadJoint.enableCollision = false;
        loadJoint.breakForce = 100000f;
    }
}
