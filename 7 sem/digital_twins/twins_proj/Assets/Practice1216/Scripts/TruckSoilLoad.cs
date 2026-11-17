using UnityEngine;

public sealed class TruckSoilLoad : MonoBehaviour
{
    public Transform bed;
    public Material soilMaterial;
    public ExcavationGround ground;
    public float Amount { get; private set; }
    private Mesh mesh;
    private GameObject visual;
    private float pourAt;

    private void Awake()
    {
        visual = new GameObject("Soil_In_Truck_Bed");
        visual.transform.SetParent(bed, false);
        mesh = new Mesh { name = "Truck_Soil_Surface" };
        visual.AddComponent<MeshFilter>().sharedMesh = mesh;
        visual.AddComponent<MeshRenderer>().sharedMaterial = soilMaterial;
        visual.SetActive(false);
    }

    public bool Receive(float volume)
    {
        if (Vector3.Angle(bed.up, Vector3.up) > 20f || Amount + volume > 7f) return false;
        Amount += volume;
        RefreshSurface();
        return true;
    }

    private void FixedUpdate()
    {
        if (Amount <= .001f || Vector3.Angle(bed.up, Vector3.up) < 24f || Time.time < pourAt) return;
        float volume = Mathf.Min(.05f, Amount);
        Amount -= volume;
        Vector3 outlet = bed.TransformPoint(new Vector3(Random.Range(-.65f, .65f), .18f, -.4f));
        SoilClod.Spawn(outlet, volume, soilMaterial, ground, transform);
        pourAt = Time.time + .055f;
        RefreshSurface();
    }

    private void RefreshSurface()
    {
        visual.SetActive(Amount > .001f);
        const int columns = 6, rows = 12;
        var vertices = new Vector3[(columns + 1) * (rows + 1)];
        var uv = new Vector2[vertices.Length];
        var triangles = new int[columns * rows * 6];
        float height = Amount / 5.8f;
        for (int z = 0; z <= rows; z++) for (int x = 0; x <= columns; x++)
        {
            int i = z * (columns + 1) + x;
            float u = (float)x / columns, v = (float)z / rows;
            float mound = .3f + .7f * Mathf.Sin(u * Mathf.PI) * Mathf.Sin(v * Mathf.PI);
            vertices[i] = new Vector3((u - .5f) * 1.96f, .1f + height * mound, .12f + v * 4.65f);
            uv[i] = new Vector2(vertices[i].x, vertices[i].z);
        }
        int index = 0;
        for (int z = 0; z < rows; z++) for (int x = 0; x < columns; x++)
        {
            int a = z * (columns + 1) + x, c = a + columns + 1;
            triangles[index++] = a; triangles[index++] = c; triangles[index++] = a + 1;
            triangles[index++] = a + 1; triangles[index++] = c; triangles[index++] = c + 1;
        }
        mesh.Clear(); mesh.vertices = vertices; mesh.uv = uv; mesh.triangles = triangles;
        mesh.RecalculateNormals(); mesh.RecalculateBounds();
    }

    private void OnDestroy()
    {
        if (mesh != null) Destroy(mesh);
    }
}
