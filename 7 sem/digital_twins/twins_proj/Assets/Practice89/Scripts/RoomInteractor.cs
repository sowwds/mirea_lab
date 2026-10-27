using UnityEngine;
using UnityEngine.InputSystem;

public sealed class RoomInteractor : MonoBehaviour
{
    public Camera view;
    public float reach = 3f;
    public TMPro.TMP_Text prompt;
    public GameObject promptBackground;
    public string CurrentPrompt { get; private set; }
    private Rigidbody held;
    private RoomMovable movable;
    private Collider[] heldColliders;
    private Collider playerCollider;
    private bool previousKinematic;
    private float dragHeight;

    private void Awake() { playerCollider = GetComponent<CharacterController>(); }
    private void Update()
    {
        UpdateTarget();
        if (Cursor.lockState != CursorLockMode.Locked || Keyboard.current == null) return;
        if (Keyboard.current.eKey.wasPressedThisFrame) Use();
    }
    public void UpdateTarget()
    {
        string text = "";
        if (held != null) text = movable.canLift ? "Положить — E" : "Отпустить — E";
        else if (Physics.Raycast(view.transform.position, view.transform.forward, out var hit, reach, ~0, QueryTriggerInteraction.Ignore))
        {
            var candle = hit.collider.GetComponentInParent<RoomCandle>();
            var target = hit.collider.GetComponentInParent<RoomMovable>();
            var action = hit.collider.GetComponentInParent<RoomAction>();
            if (action != null) text = action.Hint;
            else if (candle != null) text = candle.IsLit ? "Потушить — E" : "Зажечь — E";
            else if (target != null)
            {
                var guard = target.GetComponent<CupboardItemGuard>();
                text = guard != null && !guard.CanTake ? "Сначала откройте дверцу" : (target.canLift ? "Взять — E" : "Передвинуть — E");
            }
        }
        CurrentPrompt = text;
        if (prompt != null) prompt.text = text;
        if (promptBackground != null) promptBackground.SetActive(text.Length > 0);
    }
    public void Use()
    {
        if (held != null) { Drop(); return; }
        if (!Physics.Raycast(view.transform.position, view.transform.forward, out var hit, reach, ~0, QueryTriggerInteraction.Ignore)) return;
        var action = hit.collider.GetComponentInParent<RoomAction>();
        if (action != null) { action.Use(); UpdateTarget(); return; }
        var candle = hit.collider.GetComponentInParent<RoomCandle>();
        if (candle != null) { candle.Toggle(); UpdateTarget(); return; }
        var target = hit.collider.GetComponentInParent<RoomMovable>();
        if (target != null) Grab(target);
    }
    public void Grab(RoomMovable target)
    {
        var guard = target.GetComponent<CupboardItemGuard>();
        if (guard != null && !guard.CanTake) return;
        Drop();
        movable = target;
        held = target.GetComponent<Rigidbody>();
        previousKinematic = held.isKinematic;
        dragHeight = held.position.y;
        heldColliders = held.GetComponentsInChildren<Collider>();
        foreach (var col in heldColliders) Physics.IgnoreCollision(col, playerCollider, true);
        held.isKinematic = true;
    }
    private void FixedUpdate()
    {
        if (held == null) return;
        Vector3 target = view.transform.position + view.transform.forward * (movable.canLift ? 1.7f : 2.1f);
        if (!movable.canLift) target.y = dragHeight;
        Vector3 offset = target - held.position;
        float distance = offset.magnitude;
        if (distance > 6f) { Drop(); return; }
        if (distance < 0.01f) return;
        // Sweep the actual colliders, so held objects cannot be carried through walls.
        float travel = Mathf.Min(distance, 9f * Time.fixedDeltaTime);
        if (held.SweepTest(offset.normalized, out var hit, travel + 0.04f, QueryTriggerInteraction.Ignore))
            travel = Mathf.Max(0f, hit.distance - 0.04f);
        held.MovePosition(held.position + offset.normalized * travel);
    }
    public void Drop()
    {
        if (held == null) return;
        held.isKinematic = previousKinematic;
        foreach (var col in heldColliders)
            if (col != null && playerCollider != null) Physics.IgnoreCollision(col, playerCollider, false);
        held = null;
        movable = null;
    }
    private void OnDisable() { Drop(); }
}
