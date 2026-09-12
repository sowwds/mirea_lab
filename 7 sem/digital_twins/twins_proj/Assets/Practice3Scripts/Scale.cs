using UnityEngine;
using UnityEngine.InputSystem;

public class Scale : MonoBehaviour
{
    [SerializeField] private float scaleSpeed = 1.2f;
    [SerializeField] private float minimumScale = 0.2f;

    private void Update()
    {
        var keyboard = Keyboard.current;
        if (keyboard == null) return;

        Vector3 delta = Vector3.zero;

        // Масштабирование по осям: U/J — X, I/K — Y, O/L — Z.
        if (keyboard.uKey.isPressed) delta.x += 1f;
        if (keyboard.jKey.isPressed) delta.x -= 1f;
        if (keyboard.iKey.isPressed) delta.y += 1f;
        if (keyboard.kKey.isPressed) delta.y -= 1f;
        if (keyboard.oKey.isPressed) delta.z += 1f;
        if (keyboard.lKey.isPressed) delta.z -= 1f;

        if (delta != Vector3.zero)
        {
            Vector3 nextScale = transform.localScale + delta * scaleSpeed * Time.deltaTime;
            transform.localScale = new Vector3(
                Mathf.Max(minimumScale, nextScale.x),
                Mathf.Max(minimumScale, nextScale.y),
                Mathf.Max(minimumScale, nextScale.z));
        }
    }
}
