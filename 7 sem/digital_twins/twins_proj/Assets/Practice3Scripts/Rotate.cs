using UnityEngine;
using UnityEngine.InputSystem;

public class Rotate : MonoBehaviour
{
    [SerializeField] private float angularSpeed = 90f;

    private void Update()
    {
        var keyboard = Keyboard.current;
        if (keyboard == null) return;

        Vector3 axis = Vector3.zero;

        // X, Y и Z задают поворот вокруг соответствующих осей.
        if (keyboard.xKey.isPressed) axis.x = 1f;
        if (keyboard.yKey.isPressed) axis.y = 1f;
        if (keyboard.zKey.isPressed) axis.z = 1f;

        if (axis != Vector3.zero)
            transform.Rotate(axis * angularSpeed * Time.deltaTime, Space.Self);
    }
}
