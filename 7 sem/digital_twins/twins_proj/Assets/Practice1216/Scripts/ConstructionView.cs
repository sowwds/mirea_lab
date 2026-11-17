using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.EventSystems;
using TMPro;

public sealed class ConstructionView : MonoBehaviour
{
    public Camera sceneCamera;
    public TMP_Text selectedTitle, controls, status;
    public Transform cameraFocus;
    public ConstructionMachine Selected { get; private set; }
    private float distance = 65f;
    private readonly Vector3 direction = new Vector3(0, 36, -46).normalized;
    private Vector3 focus;
    private float orbitYaw, orbitPitch = 30f, orbitDistance = 14f;
    private Vector3 followedPosition;

    private void Start()
    {
        Cursor.lockState = CursorLockMode.None;
        Cursor.visible = true;
        focus = cameraFocus.position;
        Select(null);
        UpdateCamera();
    }
    private void Update()
    {
        var mouse = Mouse.current;
        var keyboard = Keyboard.current;
        if (mouse == null || keyboard == null) return;
        if (keyboard.escapeKey.wasPressedThisFrame) { Select(null); return; }
        bool overUI = EventSystem.current != null && EventSystem.current.IsPointerOverGameObject();
        if (Selected != null)
        {
            Cursor.lockState = CursorLockMode.Locked;
            Cursor.visible = false;
            if (Application.isFocused)
            {
                Vector2 delta = mouse.delta.ReadValue();
                orbitYaw = Mathf.Repeat(orbitYaw + delta.x * .15f, 360f);
                orbitPitch = Mathf.Clamp(orbitPitch - delta.y * .15f, 12f, 75f);
                orbitDistance = Mathf.Clamp(orbitDistance - mouse.scroll.ReadValue().y / 120f * 1.2f, 5f, 32f);
            }
        }
        else if (mouse.leftButton.wasPressedThisFrame && !overUI)
        {
            var ray = sceneCamera.ScreenPointToRay(mouse.position.ReadValue());
            Select(Physics.Raycast(ray, out var hit, 300f) ? hit.collider.GetComponentInParent<ConstructionMachine>() : null);
        }
        if (Selected == null && !overUI)
        {
            distance = Mathf.Clamp(distance - mouse.scroll.ReadValue().y / 120f * 3f, 30f, 90f);
            if (mouse.middleButton.isPressed)
            {
                Vector2 delta = mouse.delta.ReadValue();
                Vector3 right = Vector3.ProjectOnPlane(sceneCamera.transform.right, Vector3.up).normalized;
                Vector3 forward = Vector3.ProjectOnPlane(sceneCamera.transform.forward, Vector3.up).normalized;
                focus -= (right * delta.x + forward * delta.y) * distance * .0012f;
                focus.x = Mathf.Clamp(focus.x, -18, 18);
                focus.z = Mathf.Clamp(focus.z, -16, 16);
            }
        }
        if (keyboard.homeKey.wasPressedThisFrame) { Select(null); distance = 65f; focus = cameraFocus.position; }
        if (Selected != null && status != null)
        {
            var bucket = Selected.GetComponent<ExcavationBucket>();
            var truck = Selected.GetComponent<TruckSoilLoad>();
            status.text = bucket != null ? "В ковше: " + Mathf.RoundToInt(bucket.FillRatio * 100f) + "%  •  Копайте в любом месте грунтовой части площадки"
                : Selected.kind == ConstructionMachineKind.Crane
                    ? (Selected.HasLoad ? "Груз закреплён на крюке" : "Подведите крюк к петле груза и нажмите Space")
                    : truck != null ? "Земля в кузове: " + truck.Amount.ToString("F2") + " м³  •  R — поднять кузов и разгрузить"
                    : "Мышь — обзор   •   Колесо — приближение   •   Esc — выбор другой техники";
        }
    }
    public void Select(ConstructionMachine machine)
    {
        if (Selected != null) Selected.SetSelected(false);
        Selected = machine;
        if (Selected != null) Selected.SetSelected(true);
        Cursor.lockState = machine != null ? CursorLockMode.Locked : CursorLockMode.None;
        Cursor.visible = machine == null;
        if (machine != null)
        {
            orbitYaw = machine.transform.eulerAngles.y;
            orbitPitch = 30f;
            orbitDistance = machine.kind == ConstructionMachineKind.Crane ? 20f
                : machine.kind == ConstructionMachineKind.Excavator ? 16f : 12f;
            followedPosition = machine.transform.position + Vector3.up * 1.8f;
        }
        else focus = cameraFocus.position;
        selectedTitle.text = machine != null ? machine.displayName : "Выберите строительную технику";
        controls.text = machine != null ? machine.Controls + "\nМышь — обзор  •  Колесо — приближение  •  Esc — общий вид"
            : "Щёлкните по машине — камера перейдёт к ней, мышь будет вращать обзор.\nКолесо — приближение  •  Средняя кнопка мыши — сдвиг камеры  •  Home — общий вид";
        status.text = machine != null ? "Мышь — обзор  •  Esc — общий вид и свободный курсор"
            : "Свободный курсор  •  Щёлкните по технике для управления";
    }
    private void LateUpdate() => UpdateCamera();

    private void UpdateCamera()
    {
        if (Selected == null)
        {
            sceneCamera.transform.position = focus + direction * distance;
            sceneCamera.transform.LookAt(focus);
            return;
        }
        Vector3 target = Selected.transform.position + Vector3.up * 1.8f;
        followedPosition = Vector3.Lerp(followedPosition, target, 1f - Mathf.Exp(-12f * Time.deltaTime));
        Vector3 offset = Quaternion.Euler(orbitPitch, orbitYaw, 0f) * Vector3.back;
        float cameraDistance = orbitDistance;
        foreach (var hit in Physics.SphereCastAll(followedPosition, .3f, offset, orbitDistance, ~0, QueryTriggerInteraction.Ignore))
        {
            if (hit.collider.transform.IsChildOf(Selected.transform)) continue;
            cameraDistance = Mathf.Min(cameraDistance, Mathf.Max(.8f, hit.distance - .2f));
        }
        sceneCamera.transform.position = followedPosition + offset * cameraDistance;
        sceneCamera.transform.LookAt(followedPosition);
    }

    private void OnDisable()
    {
        Cursor.lockState = CursorLockMode.None;
        Cursor.visible = true;
    }
}
