# DXRNvTutorial

Implemented the Nvidia DXR tutorial with Microsoft DXR Fallback Layer, tested on Nvidia GTX 980Ti (AMD won't work). Extended nv_helpers_dx12 to work with both D3D12 prototype device and fallback device.

This project isn't updated for Windows 10 RS5 API so it won't continue to work. 

## Fallback Layer workaround

Due to the limitations of Fallback Layer, we have to use a slightly different raytracing pipeline and resource binding layout than the tutorial. 

* Bind top level acceleration structure and output UAV with global root signature rather than local root signature of a RayGen shader. That's because fallback layer uses a special `SetTopLevelAccelerationStructure()` routine to bind the acceleration structure.
* Need to bind the descriptor heap when 1) building acceleration structures and 2) calling DispatchRays(). That's for the wrapped pointer to work (see Microsoft Fallback Layer documentation).
* Do not use structured buffers in ray tracing shaders, Fallback Layer currently don't support them. Use raw buffers instead. 

## Reference

https://github.com/Microsoft/DirectX-Graphics-Samples/tree/master/Libraries/D3D12RaytracingFallback
https://github.com/Microsoft/DirectX-Graphics-Samples/tree/master/Samples/Desktop/D3D12Raytracing
https://developer.nvidia.com/rtx/raytracing/dxr/DX12-Raytracing-tutorial-Part-1
https://developer.nvidia.com/rtx/raytracing/dxr/DX12-Raytracing-tutorial-Part-2
https://github.com/NVIDIAGameWorks/DxrTutorials
http://intro-to-dxr.cwyman.org/
