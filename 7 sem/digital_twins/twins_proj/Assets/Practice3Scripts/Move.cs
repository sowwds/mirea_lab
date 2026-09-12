using UnityEngine;
using UnityEngine.InputSystem;

public class Move : MonoBehaviour
{
    [SerializeField] private float speed = 3f;

    private void Update()
    {
        var keyboard = Keyboard.current;
        if (keyboard == null) return;

        Vector3 direction = Vector3.zero;

        if (keyboard.rightArrowKey.isPressed || keyboard.dKey.isPressed) direction.x += 1f;
        if (keyboard.leftArrowKey.isPressed || keyboard.aKey.isPressed) direction.x -= 1f;
        if (keyboard.upArrowKey.isPressed || keyboard.wKey.isPressed) direction.z += 1f;
        if (keyboard.downArrowKey.isPressed || keyboard.sKey.isPressed) direction.z -= 1f;

        // Вертикальное перемещение куба: Space — вверх, Left Ctrl — вниз.
        if (keyboard.spaceKey.isPressed || keyboard.pageUpKey.isPressed) direction.y += 1f;
        if (keyboard.leftCtrlKey.isPressed || keyboard.pageDownKey.isPressed) direction.y -= 1f;

        if (direction != Vector3.zero)
            transform.position += direction.normalized * speed * Time.deltaTime;
    }
}
