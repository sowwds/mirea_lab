using UnityEngine;
using UnityEngine.UI;
using UnityEngine.EventSystems;
using System.Collections;

namespace MyGameNamespace
{
    public class OpenLinkOnButtonClick : MonoBehaviour, IPointerClickHandler
    {

        
        public string linkURL;

       
        public void OnPointerClick(PointerEventData eventData)
        {
           
            Application.OpenURL(linkURL);
        }
    }
}
