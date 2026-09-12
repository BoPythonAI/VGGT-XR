using System;
using System.IO;
using System.Reflection;
using GaussianSplatting.Editor;
using GaussianSplatting.Runtime;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;

public static class BuildDemo
{
    [Serializable] public class CameraData { public Vector3 position,forward,up; public float verticalFov; public int gaussianCount; }
    const string Package = "Packages/org.nesnausk.gaussian-splatting/Shaders/";

    [MenuItem("VGGT-XR/Create Desktop Demo Scene")]
    public static void CreateScene()
    {
        string input = Path.Combine(Application.streamingAssetsPath,"scene.ply");
        if (!File.Exists(input)) throw new FileNotFoundException("Copy scene.ply, unity_camera.json, points.ply and confidence.ply into Assets/StreamingAssets before building",input);
        Directory.CreateDirectory("Assets/GaussianAssets");Directory.CreateDirectory("Assets/Scenes");
        var creator = ScriptableObject.CreateInstance<GaussianSplatAssetCreator>();
        Type type=creator.GetType();BindingFlags flags=BindingFlags.Instance|BindingFlags.NonPublic;
        type.GetField("m_InputFile",flags).SetValue(creator,input);
        type.GetField("m_OutputFolder",flags).SetValue(creator,"Assets/GaussianAssets");
        type.GetField("m_ImportCameras",flags).SetValue(creator,false);
        // Float formats avoid import-time clustering and preserve the budgeted experiment.
        type.GetField("m_FormatPos",flags).SetValue(creator,GaussianSplatAsset.VectorFormat.Float32);
        type.GetField("m_FormatScale",flags).SetValue(creator,GaussianSplatAsset.VectorFormat.Float32);
        type.GetField("m_FormatColor",flags).SetValue(creator,GaussianSplatAsset.ColorFormat.Float32x4);
        type.GetField("m_FormatSH",flags).SetValue(creator,GaussianSplatAsset.SHFormat.Float32);
        try { type.GetMethod("CreateAsset",flags).Invoke(creator,null); }
        finally { UnityEngine.Object.DestroyImmediate(creator); }
        AssetDatabase.Refresh();
        var asset=AssetDatabase.LoadAssetAtPath<GaussianSplatAsset>("Assets/GaussianAssets/scene.asset");
        if (!asset || asset.splatCount<=0) throw new InvalidDataException("Gaussian asset import failed");
        EditorSceneManager.NewScene(NewSceneSetup.EmptyScene,NewSceneMode.Single);
        var root=new GameObject("Reconstruction");
        var gs=new GameObject("Gaussian");gs.transform.SetParent(root.transform);
        gs.transform.localScale=new Vector3(1,-1,1);gs.SetActive(false);
        var renderer=gs.AddComponent<GaussianSplatRenderer>();renderer.m_Asset=asset;
        renderer.m_ShaderSplats=AssetDatabase.LoadAssetAtPath<Shader>(Package+"RenderGaussianSplats.shader");
        renderer.m_ShaderComposite=AssetDatabase.LoadAssetAtPath<Shader>(Package+"GaussianComposite.shader");
        renderer.m_ShaderDebugPoints=AssetDatabase.LoadAssetAtPath<Shader>(Package+"GaussianDebugRenderPoints.shader");
        renderer.m_ShaderDebugBoxes=AssetDatabase.LoadAssetAtPath<Shader>(Package+"GaussianDebugRenderBoxes.shader");
        renderer.m_CSSplatUtilities=AssetDatabase.LoadAssetAtPath<ComputeShader>(Package+"SplatUtilities.compute");
        renderer.m_SHOrder=0;gs.SetActive(true);
        var geo=new GameObject("Geometry");geo.transform.SetParent(root.transform);
        var point=geo.AddComponent<PointLayer>();point.relativePath="points.ply";point.colorByConfidence=false;
        point.pointShader=AssetDatabase.LoadAssetAtPath<Shader>("Assets/Shaders/Points.shader");
        var conf=new GameObject("Confidence");conf.transform.SetParent(root.transform);
        var confPoint=conf.AddComponent<PointLayer>();confPoint.relativePath="confidence.ply";confPoint.colorByConfidence=true;
        confPoint.pointShader=point.pointShader;
        var cameraObject=new GameObject("Main Camera");cameraObject.tag="MainCamera";
        var camera=cameraObject.AddComponent<Camera>();camera.clearFlags=CameraClearFlags.SolidColor;camera.backgroundColor=Color.black;camera.nearClipPlane=.01f;camera.farClipPlane=10000;
        var data=JsonUtility.FromJson<CameraData>(File.ReadAllText(Path.Combine(Application.streamingAssetsPath,"unity_camera.json")));
        camera.transform.SetPositionAndRotation(data.position,Quaternion.LookRotation(data.forward,data.up));camera.fieldOfView=data.verticalFov;
        var controls=new GameObject("Desktop Controls").AddComponent<DesktopController>();controls.view=camera;controls.sceneRoot=root.transform;controls.gaussian=gs;controls.geometry=geo;controls.confidence=conf;controls.points=point;
        PlayerSettings.colorSpace=ColorSpace.Linear;
        PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.StandaloneWindows64,false);
        PlayerSettings.SetGraphicsAPIs(BuildTarget.StandaloneWindows64,new[]{GraphicsDeviceType.Direct3D12});
        PlayerSettings.defaultScreenWidth=1280;PlayerSettings.defaultScreenHeight=720;
        EditorSceneManager.SaveScene(UnityEngine.SceneManagement.SceneManager.GetActiveScene(),"Assets/Scenes/VGGT_XR.unity");
        EditorBuildSettings.scenes=new[]{new EditorBuildSettingsScene("Assets/Scenes/VGGT_XR.unity",true)};
        AssetDatabase.SaveAssets();Debug.Log("VGGT-XR scene created: "+asset.splatCount+" Gaussians");
    }

    [MenuItem("VGGT-XR/Build Windows Demo")]
    public static void BuildWindows()
    {
        CreateScene();
        string output=Environment.GetEnvironmentVariable("VGGTXR_UNITY_BUILD") ?? "Build/VGGT-XR.exe";
        Directory.CreateDirectory(Path.GetDirectoryName(output));
        var report=BuildPipeline.BuildPlayer(new BuildPlayerOptions { scenes=new[]{"Assets/Scenes/VGGT_XR.unity"},locationPathName=output,target=BuildTarget.StandaloneWindows64,options=BuildOptions.None });
        if(report.summary.result!=BuildResult.Succeeded)throw new Exception("Unity build failed: "+report.summary.result);
    }
}
