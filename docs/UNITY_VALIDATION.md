# Unity validation gate

The repository contains a complete project generator and keyboard/mouse controls. Runtime validation requires a locally installed, activated Unity 2022.3 Editor with Windows build support and DX12. An existing activated editor path can be passed to `scripts/desktop_demo.ps1`. Source code generation and PowerShell syntax validation do not establish Unity compilation, rendering or FPS.

Before marking the Unity item complete:

1. Generate the project on a data drive and import the exported Gaussian PLY.
2. Run the scene in Play mode: confirm Gaussian color/orientation, RGB geometry and confidence geometry refer to the same scene.
3. Hold RMB and test WASD/QE/Shift, wheel speed, +/- scale and R reset.
4. Check 1/2/3 and the GUI buttons; click a visible point and inspect its normalized reliability score.
5. Press B in each mode at the same window size; retain the generated JSON with actual GPU name, graphics API, resolution, Gaussian count, mean FPS and p95 frame time.
6. Build Windows, open the executable, repeat the display/movement checks and record 60–90 seconds of actual Unity footage.

The supplied Python video uses the gsplat CUDA renderer. It is labelled gsplat footage and is not evidence of Unity execution.

`scripts/desktop_demo.ps1` redirects Package Manager caches with `UPM_CACHE_ROOT`, which Unity 2022.3 supports. See [Unity's cache configuration documentation](https://docs.unity3d.com/2022.3/Documentation/Manual/upm-config-cache.html). Projects and Libraries must remain on a data drive. The small benchmark JSON uses Unity's persistent data directory.

The controller reports CPU allocated memory separately. GPU memory should be measured using the Unity Profiler, vendor tooling or a supported frame capture; it is not fabricated from process memory.
