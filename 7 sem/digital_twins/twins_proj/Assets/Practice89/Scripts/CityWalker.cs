using UnityEngine;
using UnityEngine.InputSystem;

[RequireComponent(typeof(CharacterController))]
public sealed class CityWalker : MonoBehaviour
{
    public Transform view;
    public float moveSpeed = 5f;
    public float sprintSpeed = 8f;
    public float jumpHeight = 1.2f;
    public float gravity = -24f;
    public float sensitivity = 0.12f;
    public Vector3 respawnPosition;
    public float PortalReadyAt { get; private set; }
    private CharacterController controller;
    private float pitch;
    private float verticalSpeed;

    private void Awake() { controller = GetComponent<CharacterController>(); }
    private void Start() { SetCursor(true); }
    private void Update()
    {
        var keyboard = Keyboard.current;
        var mouse = Mouse.current;
        if (keyboard == null) return;
        if (keyboard.escapeKey.wasPressedThisFrame) SetCursor(false);
        if (Cursor.lockState != CursorLockMode.Locked)
        {
            if (mouse != null && mouse.leftButton.wasPressedThisFrame) SetCursor(true);
            return;
        }
        if (mouse != null)
        {
            Vector2 delta = mouse.delta.ReadValue() * sensitivity;
            transform.Rotate(0f, delta.x, 0f);
            pitch = Mathf.Clamp(pitch - delta.y, -85f, 85f);
            view.localRotation = Quaternion.Euler(pitch, 0f, 0f);
        }
        Vector2 input = new Vector2(
            (keyboard.dKey.isPressed ? 1f : 0f) - (keyboard.aKey.isPressed ? 1f : 0f),
            (keyboard.wKey.isPressed ? 1f : 0f) - (keyboard.sKey.isPressed ? 1f : 0f));
        input = Vector2.ClampMagnitude(input, 1f);
        Step(input, keyboard.spaceKey.wasPressedThisFrame,
            keyboard.leftShiftKey.isPressed || keyboard.rightShiftKey.isPressed, Time.deltaTime);
        if (transform.position.y < -25f) Teleport(respawnPosition, Quaternion.identity);
    }

    // Also allows deterministic movement checks without synthesizing keyboard input.
    public void Step(Vector2 input, bool jump, bool sprint, float dt)
    {
        if (controller.isGrounded && verticalSpeed < 0f) verticalSpeed = -2f;
        if (jump && controller.isGrounded) verticalSpeed = Mathf.Sqrt(jumpHeight * -2f * gravity);
        verticalSpeed += gravity * dt;
        Vector3 planar = transform.right * input.x + transform.forward * input.y;
        controller.Move((planar * (sprint ? sprintSpeed : moveSpeed) + Vector3.up * verticalSpeed) * dt);
    }

    public void Teleport(Vector3 position, Quaternion rotation)
    {
        var interaction = GetComponent<RoomInteractor>();
        if (interaction != null) interaction.Drop();
        controller.enabled = false;
        transform.SetPositionAndRotation(position, rotation);
        controller.enabled = true;
        verticalSpeed = 0f;
        pitch = 0f;
        view.localRotation = Quaternion.identity;
        PortalReadyAt = Time.time + 1f;
        Physics.SyncTransforms();
    }

    private void OnControllerColliderHit(ControllerColliderHit hit)
    {
        var body = hit.collider.attachedRigidbody;
        if (body == null || body.isKinematic || hit.moveDirection.y < -0.3f) return;
        Vector3 direction = new Vector3(hit.moveDirection.x, 0f, hit.moveDirection.z);
        body.AddForceAtPosition(direction * 18f, hit.point, ForceMode.Force);
    }
    private static void SetCursor(bool locked)
    {
        Cursor.lockState = locked ? CursorLockMode.Locked : CursorLockMode.None;
        Cursor.visible = !locked;
    }
    private void OnDisable() { SetCursor(false); }
}
