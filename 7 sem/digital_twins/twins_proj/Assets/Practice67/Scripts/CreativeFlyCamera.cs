using UnityEngine;
using UnityEngine.InputSystem;

public sealed class CreativeFlyCamera : MonoBehaviour
{
    [Header("Movement")]
    [SerializeField, Min(0.1f)] private float moveSpeed = 18f;
    [SerializeField, Min(1f)] private float boostMultiplier = 2.5f;

    [Header("Mouse look")]
    [SerializeField, Min(0.01f)] private float mouseSensitivity = 0.12f;
    [SerializeField, Range(45f, 89f)] private float verticalLookLimit = 85f;

    private float yaw;
    private float pitch;

    private void Start()
    {
        Vector3 angles = transform.eulerAngles;
        yaw = angles.y;
        pitch = NormalizeAngle(angles.x);
        LockCursor();
    }

    private void Update()
    {
        Keyboard keyboard = Keyboard.current;
        Mouse mouse = Mouse.current;

        if (keyboard != null && keyboard.escapeKey.wasPressedThisFrame)
        {
            UnlockCursor();
            return;
        }

        if (Cursor.lockState != CursorLockMode.Locked)
        {
            if (mouse != null && mouse.leftButton.wasPressedThisFrame)
            {
                LockCursor();
            }

            return;
        }

        UpdateLook(mouse);
        UpdateMovement(keyboard);
    }

    private void UpdateLook(Mouse mouse)
    {
        if (mouse == null)
        {
            return;
        }

        Vector2 delta = mouse.delta.ReadValue();
        yaw += delta.x * mouseSensitivity;
        pitch = Mathf.Clamp(pitch - delta.y * mouseSensitivity, -verticalLookLimit, verticalLookLimit);
        transform.rotation = Quaternion.Euler(pitch, yaw, 0f);
    }

    private void UpdateMovement(Keyboard keyboard)
    {
        if (keyboard == null)
        {
            return;
        }

        float horizontal = 0f;
        float forward = 0f;
        float vertical = 0f;

        if (keyboard.aKey.isPressed) horizontal -= 1f;
        if (keyboard.dKey.isPressed) horizontal += 1f;
        if (keyboard.sKey.isPressed) forward -= 1f;
        if (keyboard.wKey.isPressed) forward += 1f;
        if (keyboard.leftShiftKey.isPressed || keyboard.rightShiftKey.isPressed) vertical -= 1f;
        if (keyboard.spaceKey.isPressed) vertical += 1f;

        Vector3 direction = transform.right * horizontal
            + transform.forward * forward
            + Vector3.up * vertical;

        if (direction.sqrMagnitude > 1f)
        {
            direction.Normalize();
        }

        bool boost = keyboard.leftCtrlKey.isPressed || keyboard.rightCtrlKey.isPressed;
        float speed = moveSpeed * (boost ? boostMultiplier : 1f);
        transform.position += direction * speed * Time.unscaledDeltaTime;
    }

    private static float NormalizeAngle(float angle)
    {
        return angle > 180f ? angle - 360f : angle;
    }

    private static void LockCursor()
    {
        Cursor.lockState = CursorLockMode.Locked;
        Cursor.visible = false;
    }

    private static void UnlockCursor()
    {
        Cursor.lockState = CursorLockMode.None;
        Cursor.visible = true;
    }

    private void OnDisable()
    {
        UnlockCursor();
    }
}
