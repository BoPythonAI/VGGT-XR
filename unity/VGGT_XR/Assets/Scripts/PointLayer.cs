using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Rendering;

public sealed class PointLayer : MonoBehaviour
{
    public string relativePath;
    public bool colorByConfidence;
    public Shader pointShader;
    private Vector3[] vertices;
    private float[] confidence;
    private Mesh mesh;
    private Material material;

    void Start() { EnsureLoaded(); }
    public void EnsureLoaded() { if (vertices == null) Load(Path.Combine(Application.streamingAssetsPath,relativePath),colorByConfidence); }

    public void Load(string path, bool colorByConfidence)
    {
        using (var file = File.OpenRead(path))
        {
            var line = new List<byte>();
            var fields = new List<string>();
            var types = new List<string>();
            int count = 0;
            string format = "";
            bool vertex = false;
            while (true)
            {
                line.Clear(); int b;
                while ((b = file.ReadByte()) != 10 && b >= 0) line.Add((byte)b);
                if (b < 0) throw new InvalidDataException("Missing PLY header");
                string text = Encoding.ASCII.GetString(line.ToArray()).Trim();
                if (text.StartsWith("format ")) format = text;
                if (text.StartsWith("element vertex ")) { count = int.Parse(text.Split(' ')[2], CultureInfo.InvariantCulture); vertex = true; }
                else if (text.StartsWith("element ")) vertex = false;
                if (vertex && text.StartsWith("property ")) { var parts = text.Split(' '); types.Add(parts[1]); fields.Add(parts[2]); }
                if (text == "end_header") break;
            }
            if (format != "format binary_little_endian 1.0" || count <= 0 || count > 500000) throw new InvalidDataException("Expected bounded binary little-endian point PLY");
            vertices = new Vector3[count]; confidence = new float[count];
            var colors = new Color[count];
            using (var reader = new BinaryReader(file, Encoding.ASCII, true))
            {
                for (int i = 0; i < count; i++)
                {
                    float x=0,y=0,z=0,r=0,g=0,bl=0,c=0;
                    for (int j = 0; j < fields.Count; j++)
                    {
                        float value = types[j] == "float" ? reader.ReadSingle() : types[j] == "uchar" ? reader.ReadByte() : throw new InvalidDataException("Unsupported PLY scalar " + types[j]);
                        switch (fields[j]) { case "x": x=value; break; case "y": y=value; break; case "z": z=value; break; case "red": r=value/255; break; case "green": g=value/255; break; case "blue": bl=value/255; break; case "confidence": c=value; break; }
                    }
                    vertices[i] = new Vector3(x,-y,z);
                    confidence[i] = c;
                    colors[i] = colorByConfidence ? new Color(1-c,c,0) : new Color(r,g,bl);
                }
            }
            mesh = new Mesh { name = "VGGT points", indexFormat = IndexFormat.UInt32 };
            mesh.vertices=vertices;mesh.colors=colors;
            var indices=new int[count];for(int i=0;i<count;i++)indices[i]=i;
            mesh.SetIndices(indices,MeshTopology.Points,0);mesh.RecalculateBounds();
            gameObject.AddComponent<MeshFilter>().sharedMesh=mesh;
            material=new Material(pointShader ? pointShader : Shader.Find("VGGTXR/Points"));material.SetFloat("_PointPixels",3f);
            gameObject.AddComponent<MeshRenderer>().sharedMaterial=material;
        }
    }

    public bool TryPick(Ray ray,out float score,out float distance)
    {
        EnsureLoaded();
        score=0;distance=float.PositiveInfinity;bool found=false;
        for(int i=0;i<vertices.Length;i++)
        {
            Vector3 p=transform.TransformPoint(vertices[i]);Vector3 v=p-ray.origin;
            float along=Vector3.Dot(v,ray.direction);if(along<=0 || along>=distance)continue;
            float perpendicular=(v-along*ray.direction).magnitude;
            if(perpendicular<Mathf.Max(.02f,along*.01f)){found=true;distance=along;score=confidence[i];}
        }
        return found;
    }

    void OnDestroy(){if(mesh)Destroy(mesh);if(material)Destroy(material);}
}
