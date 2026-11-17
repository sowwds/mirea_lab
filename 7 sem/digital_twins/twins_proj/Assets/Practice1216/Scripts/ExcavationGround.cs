using UnityEngine;
using System.Collections.Generic;

[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer), typeof(MeshCollider))]
public sealed class ExcavationGround : MonoBehaviour
{
    public int columns = 90;
    public int rows = 72;
    public float width = 54f;
    public float depth = 44f;
    public float maximumDepth = 2.8f;
    private Mesh mesh;
    private Vector3[] vertices;
    private float[] heights;
    private bool changed;
    private float refreshAt;
    private float CellArea => width / columns * depth / rows;

    public void BuildInitial()
    {
        mesh = new Mesh { name = "ExcavatableGround" };
        vertices = new Vector3[(columns + 1) * (rows + 1)];
        heights = new float[vertices.Length];
        for (int z = 0; z <= rows; z++)
            for (int x = 0; x <= columns; x++)
                vertices[z * (columns + 1) + x] = new Vector3(-width * .5f + x * width / columns, 0, -depth * .5f + z * depth / rows);
        GetComponent<MeshFilter>().sharedMesh = mesh;
        Rebuild();
    }
    private void Awake()
    {
        var source = GetComponent<MeshFilter>().sharedMesh;
        if (source == null) { BuildInitial(); return; }
        mesh = Instantiate(source);
        GetComponent<MeshFilter>().sharedMesh = mesh;
        vertices = mesh.vertices;
        heights = new float[vertices.Length];
        for (int i = 0; i < vertices.Length; i++) heights[i] = vertices[i].y;
        GetComponent<MeshCollider>().sharedMesh = mesh;
    }
    public float HeightAt(Vector3 world)
    {
        Vector3 p = transform.InverseTransformPoint(world);
        float fx = Mathf.Clamp((p.x / width + .5f) * columns, 0, columns - .001f);
        float fz = Mathf.Clamp((p.z / depth + .5f) * rows, 0, rows - .001f);
        int x = Mathf.FloorToInt(fx), z = Mathf.FloorToInt(fz), i = z * (columns + 1) + x;
        float h = Mathf.Lerp(Mathf.Lerp(heights[i], heights[i + 1], fx - x),
            Mathf.Lerp(heights[i + columns + 1], heights[i + columns + 2], fx - x), fz - z);
        return transform.position.y + h;
    }
    public float Dig(Vector3 position, float volume, float radius)
    {
        if (Mathf.Abs(position.x) > 24f || position.z < -13f || position.z > 19f) return 0f;
        return Sculpt(position, -volume, radius);
    }
    public void Deposit(Vector3 position, float volume)
    {
        Sculpt(position, volume, .95f);
    }
    private float Sculpt(Vector3 world, float volume, float radius)
    {
        if (heights == null) return 0f;
        Vector3 p = transform.InverseTransformPoint(world);
        float total = 0f;
        for (int i = 0; i < vertices.Length; i++)
        {
            float d = new Vector2(vertices[i].x - p.x, vertices[i].z - p.z).magnitude;
            if (d < radius) total += (1f - d * d / (radius * radius)) * CellArea;
        }
        if (total < .001f) return 0f;
        float actual = 0f;
        for (int i = 0; i < vertices.Length; i++)
        {
            float d = new Vector2(vertices[i].x - p.x, vertices[i].z - p.z).magnitude;
            if (d >= radius) continue;
            float delta = volume * (1f - d * d / (radius * radius)) / total;
            float next = Mathf.Clamp(heights[i] + delta, -maximumDepth, 4f);
            actual += Mathf.Abs(next - heights[i]) * CellArea;
            heights[i] = next;
        }
        changed = true;
        return actual;
    }
    private void FixedUpdate()
    {
        if (changed && Time.time >= refreshAt) { Rebuild(); refreshAt = Time.time + .12f; }
    }
    private void Rebuild()
    {
        for (int i = 0; i < vertices.Length; i++) vertices[i].y = heights[i];
        mesh.vertices = vertices;
        var uv = new Vector2[vertices.Length];
        for (int i = 0; i < uv.Length; i++) uv[i] = new Vector2(vertices[i].x, vertices[i].z);
        mesh.uv = uv;
        var triangles = new[] { new List<int>(), new List<int>(), new List<int>() };
        for (int z = 0; z < rows; z++) for (int x = 0; x < columns; x++)
        {
            int a = z * (columns + 1) + x, b = a + 1, c = a + columns + 1, d = c + 1;
            AddTriangle(triangles, a, c, b); AddTriangle(triangles, b, c, d);
        }
        mesh.subMeshCount = 3;
        for (int i = 0; i < 3; i++) mesh.SetTriangles(triangles[i], i);
        mesh.RecalculateNormals(); mesh.RecalculateBounds();
        var collider = GetComponent<MeshCollider>(); collider.sharedMesh = null; collider.sharedMesh = mesh;
        changed = false;
    }
    private void AddTriangle(List<int>[] groups, int a, int b, int c)
    {
        float h = (heights[a] + heights[b] + heights[c]) / 3f;
        var target = groups[h < -.08f ? 1 : h > .08f ? 2 : 0];
        target.Add(a); target.Add(b); target.Add(c);
    }
}
