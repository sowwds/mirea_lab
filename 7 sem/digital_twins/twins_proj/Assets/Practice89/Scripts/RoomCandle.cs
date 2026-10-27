using UnityEngine;

public sealed class RoomCandle : MonoBehaviour
{
    public Light candleLight;
    public GameObject flame;
    public bool startsLit = true;
    public bool IsLit { get; private set; }
    private float baseIntensity;

    private void Awake()
    {
        baseIntensity = candleLight != null ? candleLight.intensity : 1f;
        SetLit(startsLit);
    }
    public void Toggle() { SetLit(!IsLit); }
    public void SetLit(bool lit)
    {
        IsLit = lit;
        if (candleLight != null) candleLight.enabled = lit;
        if (flame != null) flame.SetActive(lit);
    }
    private void Update()
    {
        if (IsLit && candleLight != null)
            candleLight.intensity = baseIntensity * (0.93f + 0.07f * Mathf.PerlinNoise(Time.time * 5f, transform.position.x));
    }
}
