using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Profiling;

public sealed class DesktopController : MonoBehaviour
{
    public Camera view;
    public Transform sceneRoot;
    public GameObject gaussian, geometry, confidence;
    public PointLayer points;
    public float speed = 1f;
    private Vector3 initialPosition;
    private Quaternion initialRotation;
    private string mode = "Gaussian", selection = "Click geometry to inspect confidence";
    private float smoothedDelta;
    private List<float> frameTimes;
    private int warmup;

    void Start()
    {
        initialPosition = view.transform.position;
        initialRotation = view.transform.rotation;
        QualitySettings.vSyncCount = 0;
        Application.targetFrameRate = -1;
        SetMode(1);
    }

    public void SetMode(int value)
    {
        gaussian.SetActive(value == 1);
        geometry.SetActive(value == 2);
        confidence.SetActive(value == 3);
        mode = value == 1 ? "Gaussian" : value == 2 ? "Geometry" : "Confidence";
    }

    void Update()
    {
        smoothedDelta = Mathf.Lerp(smoothedDelta, Time.unscaledDeltaTime, .1f);
        if (Input.GetMouseButton(1))
        {
            Vector3 e = view.transform.eulerAngles;
            float pitch = e.x > 180 ? e.x - 360 : e.x;
            pitch = Mathf.Clamp(pitch - Input.GetAxis("Mouse Y") * 2, -89, 89);
            view.transform.rotation = Quaternion.Euler(pitch, e.y + Input.GetAxis("Mouse X") * 2, 0);
            Vector3 motion = Vector3.zero;
            if (Input.GetKey(KeyCode.W)) motion += view.transform.forward;
            if (Input.GetKey(KeyCode.S)) motion -= view.transform.forward;
            if (Input.GetKey(KeyCode.D)) motion += view.transform.right;
            if (Input.GetKey(KeyCode.A)) motion -= view.transform.right;
            if (Input.GetKey(KeyCode.E)) motion += Vector3.up;
            if (Input.GetKey(KeyCode.Q)) motion -= Vector3.up;
            view.transform.position += motion * speed * (Input.GetKey(KeyCode.LeftShift) ? 3 : 1) * Time.deltaTime;
        }
        speed *= Mathf.Exp(Input.mouseScrollDelta.y * .1f);
        speed = Mathf.Clamp(speed, .02f, 100);
        if (Input.GetKeyDown(KeyCode.Alpha1)) SetMode(1);
        if (Input.GetKeyDown(KeyCode.Alpha2)) SetMode(2);
        if (Input.GetKeyDown(KeyCode.Alpha3)) SetMode(3);
        if (Input.GetKeyDown(KeyCode.R))
        {
            view.transform.SetPositionAndRotation(initialPosition, initialRotation);
            sceneRoot.localScale = Vector3.one;
        }
        float scaling = Input.GetKey(KeyCode.Equals) || Input.GetKey(KeyCode.KeypadPlus) ? 1 : Input.GetKey(KeyCode.Minus) || Input.GetKey(KeyCode.KeypadMinus) ? -1 : 0;
        sceneRoot.localScale *= Mathf.Exp(scaling * Time.deltaTime);
        if (Input.GetMouseButtonDown(0) && Input.mousePosition.x > 380)
        {
            Ray ray = view.ScreenPointToRay(Input.mousePosition);
            if (points.TryPick(ray, out float score, out float distance))
                selection = $"Normalized confidence: {score:F2}\nDistance: {distance:F2} reconstruction units";
            else selection = "No geometry near the ray";
        }
        if (Input.GetKeyDown(KeyCode.B)) { frameTimes = new List<float>(); warmup = 60; }
        if (frameTimes != null)
        {
            if (warmup > 0) warmup--;
            else frameTimes.Add(Time.unscaledDeltaTime * 1000);
            if (frameTimes.Count >= 600) SaveBenchmark();
        }
    }

    [Serializable] public class Benchmark
    {
        public string mode, gpu, graphicsApi, unityVersion, timestamp;
        public int width, height, frames, gaussianCount;
        public float meanFps, medianFrameMs, p95FrameMs, scale;
        public long allocatedCpuBytes;
    }

    void SaveBenchmark()
    {
        frameTimes.Sort();
        float sum = 0; foreach (float x in frameTimes) sum += x;
        var renderer = gaussian.GetComponent<GaussianSplatting.Runtime.GaussianSplatRenderer>();
        var row = new Benchmark { mode = mode, gpu = SystemInfo.graphicsDeviceName, graphicsApi = SystemInfo.graphicsDeviceType.ToString(), unityVersion = Application.unityVersion, timestamp = DateTime.UtcNow.ToString("o"), width = Screen.width, height = Screen.height, frames = frameTimes.Count, gaussianCount = renderer.m_Asset.splatCount, meanFps = 1000f / (sum / frameTimes.Count), medianFrameMs = frameTimes[frameTimes.Count / 2], p95FrameMs = frameTimes[(int)(frameTimes.Count * .95f)], allocatedCpuBytes = Profiler.GetTotalAllocatedMemoryLong(), scale = sceneRoot.localScale.x };
        string path = Path.Combine(Application.persistentDataPath, "benchmark_" + DateTime.UtcNow.ToString("yyyyMMdd_HHmmss") + ".json");
        File.WriteAllText(path, JsonUtility.ToJson(row, true));
        selection = "Benchmark saved:\n" + path;
        Debug.Log(selection);
        frameTimes = null;
    }

    void OnGUI()
    {
        GUI.Box(new Rect(12, 12, 350, 245), "VGGT-XR Desktop Demo");
        GUI.Label(new Rect(25, 40, 330, 25), $"{mode} | {1 / Mathf.Max(smoothedDelta, .00001f):F0} FPS");
        if (GUI.Button(new Rect(25, 70, 100, 28), "1 Gaussian")) SetMode(1);
        if (GUI.Button(new Rect(130, 70, 100, 28), "2 Geometry")) SetMode(2);
        if (GUI.Button(new Rect(235, 70, 105, 28), "3 Confidence")) SetMode(3);
        GUI.Label(new Rect(25, 108, 325, 60), "Hold RMB + WASD / QE: fly; Shift: faster\nWheel: speed; +/-: scale; R: reset; B: benchmark\nConfidence is a relative VGGT score, not probability.");
        GUI.Label(new Rect(25, 174, 325, 72), frameTimes == null ? selection : $"Benchmarking {mode}: {frameTimes.Count}/600 frames");
    }
}
