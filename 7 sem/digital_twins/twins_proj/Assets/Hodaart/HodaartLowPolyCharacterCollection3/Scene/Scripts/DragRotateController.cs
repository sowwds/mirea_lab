using UnityEngine;
using UnityEngine.EventSystems;


namespace LowPolyCharacterCollection3
{
    public class DragRotateController : MonoBehaviour,
        IPointerDownHandler, IDragHandler, IPointerUpHandler
    {
        [SerializeField] private Transform targetToRotate;
        [SerializeField] private float rotationSpeed = 0.3f;
        [SerializeField] private float smoothSpeed = 10f;

        private float targetY;
        private float currentY;
        private float lastPointerX;
        private bool isDragging;

        void Start()
        {
            currentY = targetY = targetToRotate.eulerAngles.y;
        }

        void Update()
        {
            currentY = Mathf.Lerp(currentY, targetY, Time.deltaTime * smoothSpeed);
            targetToRotate.rotation = Quaternion.Euler(0f, currentY, 0f);
        }

        public void OnPointerDown(PointerEventData eventData)
        {
            isDragging = true;
            lastPointerX = eventData.position.x;
        }

        public void OnDrag(PointerEventData eventData)
        {
            if (!isDragging) return;

            float deltaX = eventData.position.x - lastPointerX;
            lastPointerX = eventData.position.x;

            targetY -= deltaX * rotationSpeed;
        }

        public void OnPointerUp(PointerEventData eventData)
        {
            isDragging = false;
        }
    }
}
