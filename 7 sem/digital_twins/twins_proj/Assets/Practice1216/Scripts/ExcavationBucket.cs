using System.Collections;
using UnityEngine;
using UnityEngine.InputSystem;

public sealed class ExcavationBucket : MonoBehaviour
{
    public ExcavationGround ground;
    public Transform bucket;
    public Transform soilVisual;
    public Material soilMaterial;
    public float capacity = 1.2f;
    public float Amount { get; private set; }
    public float FillRatio => Amount / capacity;
    private ConstructionMachine machine;
    private Renderer bucketRenderer;
    private Vector3 previousPosition;
    private float nextDig;

    private void Awake()
    {
        machine = GetComponent<ConstructionMachine>();
        bucketRenderer = bucket.GetComponent<Renderer>();
        previousPosition = bucket.position;
        UpdateSoil();
    }
    private void Update()
    {
        if (!machine.IsSelected || Keyboard.current == null) return;
        if (Keyboard.current.spaceKey.wasPressedThisFrame) Dump();
    }
    private void FixedUpdate()
    {
        float movement = Vector3.Distance(previousPosition, bucket.position);
        previousPosition = bucket.position;
        if (!machine.IsSelected || Keyboard.current == null || Time.time < nextDig) return;
        var k = Keyboard.current;
        bool controls = k.fKey.isPressed || k.gKey.isPressed || k.hKey.isPressed || k.wKey.isPressed || k.sKey.isPressed;
        if (k.bKey.isPressed || (controls && movement > .002f))
        {
            Dig(.18f);
            nextDig = Time.time + .15f;
        }
    }
    public float Dig(float dt)
    {
        if (ground == null || Amount >= capacity - .001f) return 0f;
        Bounds b = bucketRenderer.bounds;
        Vector3 point = b.center;
        if (b.min.y > ground.HeightAt(point) + .36f) return 0f;
        float removed = ground.Dig(point, Mathf.Min(capacity - Amount, dt * .8f), .95f);
        Amount += removed;
        UpdateSoil();
        return removed;
    }
    public void Dump()
    {
        if (Amount < .01f) return;
        var b = bucketRenderer.bounds;
        Vector3 drop = b.center;
        drop.y = Mathf.Max(b.min.y + .2f, ground.HeightAt(drop) + .45f);
        int count = Mathf.Clamp(Mathf.CeilToInt(Amount / .035f), 3, 36);
        float each = Amount / count;
        Amount = 0f;
        UpdateSoil();
        StartCoroutine(Pour(drop, count, each));
    }
    private IEnumerator Pour(Vector3 drop, int count, float each)
    {
        for (int i = 0; i < count; i++)
        {
            SoilClod.Spawn(drop + new Vector3(Random.Range(-.12f, .12f), .12f, Random.Range(-.12f, .12f)),
                each, soilMaterial, ground, machine.transform);
            yield return new WaitForSeconds(.07f);
        }
    }
    private void UpdateSoil()
    {
        if (soilVisual == null) return;
        soilVisual.gameObject.SetActive(Amount > .01f);
        soilVisual.localScale = new Vector3(1.05f, Mathf.Lerp(.08f, .6f, FillRatio), 1.25f);
    }
}
