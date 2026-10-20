using UnityEngine;
using UnityEngine.EventSystems;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem.UI;
#endif

namespace LowPolyCharacterCollection3
{
    [DisallowMultipleComponent]
    public class EventSystemEnsurer : MonoBehaviour
    {
        private void Awake()
        {
            if (FindObjectOfType<EventSystem>() == null)
            {
                var go = new GameObject("EventSystem");
                go.AddComponent<EventSystem>();
#if ENABLE_INPUT_SYSTEM && !ENABLE_LEGACY_INPUT_MANAGER
                go.AddComponent<InputSystemUIInputModule>();
                Debug.Log("[EventSystemEnsurer] No EventSystem found — created one with InputSystemUIInputModule.");
#else
                go.AddComponent<StandaloneInputModule>();
                Debug.Log("[EventSystemEnsurer] No EventSystem found — created one with StandaloneInputModule.");
#endif
            }
        }
    }
}