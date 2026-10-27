using UnityEngine;

public abstract class RoomAction : MonoBehaviour
{
    public abstract string Hint { get; }
    public abstract void Use();
}
