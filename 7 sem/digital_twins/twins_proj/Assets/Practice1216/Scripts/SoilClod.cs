using UnityEngine;

public sealed class SoilClod : MonoBehaviour
{
    public ExcavationGround ground;
    public float volume;
    private bool deposited;
    public static SoilClod Spawn(Vector3 position, float volume, Material material, ExcavationGround ground, Transform ignore)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        go.name = "Excavated_Soil";
        go.transform.position = position;
        go.transform.localScale = Vector3.one * .18f;
        go.GetComponent<Renderer>().sharedMaterial = material;
        var rb = go.AddComponent<Rigidbody>();
        rb.mass = .9f;
        rb.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;
        rb.interpolation = RigidbodyInterpolation.Interpolate;
        rb.linearVelocity = Vector3.down * 3f;
        if (ignore != null) foreach (var own in ignore.GetComponentsInChildren<Collider>())
            Physics.IgnoreCollision(go.GetComponent<Collider>(), own);
        go.AddComponent<LooseMaterial>();
        var clod = go.AddComponent<SoilClod>();
        clod.volume = volume;
        clod.ground = ground;
        return clod;
    }
    private void OnCollisionEnter(Collision collision)
    {
        if (deposited) return;
        var truck = collision.collider.GetComponentInParent<TruckSoilLoad>();
        if (truck != null && collision.collider.transform.IsChildOf(truck.bed)
            && truck.Receive(volume))
        {
            deposited = true;
            Destroy(gameObject);
        }
        else if (collision.collider.GetComponent<ExcavationGround>() != null)
        {
            deposited = true;
            ground.Deposit(collision.GetContact(0).point, volume);
            Destroy(gameObject);
        }
    }
}
