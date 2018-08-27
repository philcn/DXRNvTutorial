//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

#include "stdafx.h"
#include "DXRNvTutorialApp.h"
#include "DirectXRaytracingHelper.h"

#include "nv_helpers_dx12/DXRHelper.h"
#include "nv_helpers_dx12/BottomLevelASGenerator.h"
#include "nv_helpers_dx12/TopLevelASGenerator.h"
#include "nv_helpers_dx12/RootSignatureGenerator.h"
#include "nv_helpers_dx12/RaytracingPipelineGenerator.h"   
#include "nv_helpers_dx12/ShaderBindingTableGenerator.h"

#include "CompiledShaders/Raytracing.hlsl.h"

using namespace std;
using namespace DX;

DXRNvTutorialApp::DXRNvTutorialApp(UINT width, UINT height, std::wstring name) :
    DXSample(width, height, name),
    mUseDXRDriver(false),
	mRaytracingEnabled(true),
	mDescriptorsAllocated(0)
{
    UpdateForSizeChange(width, height);
}

void DXRNvTutorialApp::EnableDXRExperimentalFeatures(IDXGIAdapter1* adapter)
{
    // DXR is an experimental feature and needs to be enabled before creating a D3D12 device.
    mUseDXRDriver = EnableRaytracing(adapter);

    if (!mUseDXRDriver) {
        OutputDebugString(
            L"Could not enable raytracing driver (D3D12EnableExperimentalFeatures() failed).\n" \
            L"Possible reasons:\n" \
            L"  1) your OS is not in developer mode.\n" \
            L"  2) your GPU driver doesn't match the D3D12 runtime loaded by the app (d3d12.dll and friends).\n" \
            L"  3) your D3D12 runtime doesn't match the D3D12 headers used by your app (in particular, the GUID passed to D3D12EnableExperimentalFeatures).\n\n");

        OutputDebugString(L"Enabling compute based fallback raytracing support.\n");
        ThrowIfFalse(EnableComputeRaytracingFallback(adapter), L"Could not enable compute based fallback raytracing support (D3D12EnableExperimentalFeatures() failed).\n");
    }
}

void DXRNvTutorialApp::OnInit()
{
    m_deviceResources = std::make_unique<DeviceResources>(
        DXGI_FORMAT_R8G8B8A8_UNORM,
        DXGI_FORMAT_UNKNOWN,
        FrameCount,
        D3D_FEATURE_LEVEL_12_0,
        // Sample shows handling of use cases with tearing support, which is OS dependent and has been supported since TH2.
        // Since the Fallback Layer requires Fall Creator's update (RS3), we don't need to handle non-tearing cases.
        DeviceResources::c_RequireTearingSupport,
        m_adapterIDoverride
    );
    m_deviceResources->RegisterDeviceNotify(this);
    m_deviceResources->SetWindow(Win32Application::GetHwnd(), m_width, m_height);
    m_deviceResources->InitializeDXGIAdapter();
    EnableDXRExperimentalFeatures(m_deviceResources->GetAdapter());

    m_deviceResources->CreateDeviceResources();
    m_deviceResources->CreateWindowSizeDependentResources();

    InitializeScene();

    CreateDeviceDependentResources();
    CreateWindowSizeDependentResources();
}

// Initialize scene rendering parameters.
void DXRNvTutorialApp::InitializeScene()
{
    auto frameIndex = m_deviceResources->GetCurrentFrameIndex();
}

void DXRNvTutorialApp::CreateDeviceDependentResources()
{
    // Create raytracing interfaces: raytracing device and commandlist.
    CreateRaytracingInterfaces();

    // Create root signatures for the shaders.
//	mGlobalRootSignature = CreateGlobalRootSignature();
	CreateRootSignatures();

    // Create a raytracing pipeline state object which defines the binding of shaders, state and resources to be used during raytracing.
//	CreateRaytracingPipeline();
	CreateRaytracingPipelineStateObject();

    // Create a heap for descriptors.
    CreateDescriptorHeap();

    // Build geometry to be used in the sample.
	CreateGeometries();

    // Build raytracing acceleration structures from the generated geometry.
	CreateAccelerationStructures();

    // Create constant buffers for the geometry and the scene.

    // Build shader tables, which define shaders and their local root arguments.
//	CreateShaderBindingTable(); 
	BuildShaderTables();

    InitShaderResourceHeap();
}

// Create raytracing device and command list.
void DXRNvTutorialApp::CreateRaytracingInterfaces()
{
    auto device = m_deviceResources->GetD3DDevice();
    auto commandList = m_deviceResources->GetCommandList();

	// Only support fallback layer
    if (!mUseDXRDriver) {
        CreateRaytracingFallbackDeviceFlags createDeviceFlags = CreateRaytracingFallbackDeviceFlags::None;
        ThrowIfFailed(D3D12CreateRaytracingFallbackDevice(device, createDeviceFlags, 0, IID_PPV_ARGS(&mFallbackDevice)));
        mFallbackDevice->QueryRaytracingCommandList(commandList, IID_PPV_ARGS(&mFallbackCommandList));
    } else { // DirectX Raytracing
		assert(0);
	}
}

void DXRNvTutorialApp::CreateDescriptorHeap()
{
	auto device = m_deviceResources->GetD3DDevice();

	D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc = {};
	// Allocate a heap for 3 descriptors:
	// 1 - raytracing output texture SRV
	// 2 - bottom and top level acceleration structure fallback wrapped pointer UAVs
	// /* 2 - vertex and index buffer SRVs */
	descriptorHeapDesc.NumDescriptors = 3; 
	descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	descriptorHeapDesc.NodeMask = 0;
	device->CreateDescriptorHeap(&descriptorHeapDesc, IID_PPV_ARGS(&mDescriptorHeap));
	NAME_D3D12_OBJECT(mDescriptorHeap);

	mDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
}

void DXRNvTutorialApp::CreateGeometries()
{
	Vertex triangleVertices[] =
	{
		{ { 0.0f, 0.25f, 0.0f }, { 0.0f, 0.0f, 1.0f } },
		{ { 0.25f, -0.25f, 0.0f }, { 0.0f, 0.0f, 1.0f } },
		{ { -0.25f, -0.25f, 0.0f }, { 0.0f, 0.0f, 1.0f } }
	};

	const UINT vertexBufferSize = sizeof(triangleVertices);

	auto device = m_deviceResources->GetD3DDevice();
	// Note: using upload heaps to transfer static data like vert buffers is not 
	// recommended. Every time the GPU needs it, the upload heap will be marshalled 
	// over. Please read up on Default Heap usage. An upload heap is used here for 
	// code simplicity and because there are very few verts to actually transfer.
	ThrowIfFailed(device->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
		D3D12_HEAP_FLAG_NONE,
		&CD3DX12_RESOURCE_DESC::Buffer(vertexBufferSize),
		D3D12_RESOURCE_STATE_GENERIC_READ,
		nullptr,
		IID_PPV_ARGS(&mVertexBuffer)));

	// Copy the triangle data to the vertex buffer.
	UINT8* pVertexDataBegin;
	CD3DX12_RANGE readRange(0, 0);		// We do not intend to read from this resource on the CPU.
	ThrowIfFailed(mVertexBuffer->Map(0, &readRange, reinterpret_cast<void**>(&pVertexDataBegin)));
	memcpy(pVertexDataBegin, triangleVertices, sizeof(triangleVertices));
	mVertexBuffer->Unmap(0, nullptr);
}

AccelerationStructure DXRNvTutorialApp::CreateBottomLevelAS(std::vector<std::pair<ComPtr<ID3D12Resource>, uint32_t>> vertexBuffers)
{
	nv_helpers_dx12::BottomLevelASGenerator blasGenerator;

	for (const auto &vb : vertexBuffers) {
		blasGenerator.AddVertexBuffer(vb.first.Get(), 0, vb.second, sizeof(Vertex), nullptr, 0);
	}

	UINT64 scratchSizeInBytes = 0;
	UINT64 resultSizeInBytes = 0;
	blasGenerator.ComputeASBufferSizes(mFallbackDevice.Get(), false, &scratchSizeInBytes, &resultSizeInBytes);

	auto device = m_deviceResources->GetD3DDevice();
	AccelerationStructure buffers;

	buffers.mScratch = nv_helpers_dx12::CreateBuffer(
		device, scratchSizeInBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, 
		D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nv_helpers_dx12::kDefaultHeapProps);

	D3D12_RESOURCE_STATES initialResourceState = mFallbackDevice->GetAccelerationStructureResourceState();
	buffers.mResult = nv_helpers_dx12::CreateBuffer(
		device, resultSizeInBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, 
		initialResourceState, nv_helpers_dx12::kDefaultHeapProps);

	auto commandList = m_deviceResources->GetCommandList();
	blasGenerator.Generate(commandList, mFallbackCommandList.Get(), buffers.mScratch.Get(), buffers.mResult.Get());

	return buffers;
}

AccelerationStructure DXRNvTutorialApp::CreateTopLevelAS(const std::vector<std::pair<ComPtr<ID3D12Resource>, DirectX::XMMATRIX>> &instances, UINT &outResultSizeBytes)
{
	nv_helpers_dx12::TopLevelASGenerator tlasGenerator;

	for (int i = 0; i < instances.size(); ++i) {
		tlasGenerator.AddInstance(instances[i].first.Get(), instances[i].second, i, 0);
	}

	UINT64 scratchSizeInBytes = 0;
	UINT64 resultSizeInBytes = 0;
	UINT64 instanceDescsSize = 0;
	tlasGenerator.ComputeASBufferSizes(mFallbackDevice.Get(), true, &scratchSizeInBytes, &resultSizeInBytes, &instanceDescsSize);

	auto device = m_deviceResources->GetD3DDevice();
	AccelerationStructure buffers;
	
	// Allocate on default heap since the build is done on GPU
	buffers.mScratch = nv_helpers_dx12::CreateBuffer(
		device, scratchSizeInBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, 
		D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nv_helpers_dx12::kDefaultHeapProps);

	D3D12_RESOURCE_STATES initialResourceState = mFallbackDevice->GetAccelerationStructureResourceState();
	buffers.mResult = nv_helpers_dx12::CreateBuffer(
		device, resultSizeInBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, 
		initialResourceState, nv_helpers_dx12::kDefaultHeapProps);
	
	buffers.mInstanceDesc = nv_helpers_dx12::CreateBuffer(
		device, instanceDescsSize, D3D12_RESOURCE_FLAG_NONE, 
		D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps); 

	auto commandList = m_deviceResources->GetCommandList();
	tlasGenerator.Generate(commandList, mFallbackCommandList.Get(), buffers.mScratch.Get(), buffers.mResult.Get(), buffers.mInstanceDesc.Get(), 
		device, mFallbackDevice.Get(), mDescriptorHeap.Get(), mDescriptorsAllocated, mDescriptorSize);

	outResultSizeBytes = resultSizeInBytes;

	return buffers;
}

void DXRNvTutorialApp::CreateAccelerationStructures()
{
	auto device = m_deviceResources->GetD3DDevice();
	auto commandList = m_deviceResources->GetCommandList();
	auto commandQueue = m_deviceResources->GetCommandQueue();
	auto commandAllocator = m_deviceResources->GetCommandAllocator();

	// Reset the command list for the acceleration structure construction.
	commandList->Reset(commandAllocator, nullptr);

	// Set the descriptor heaps to be used during acceleration structure build for the Fallback Layer.
	ID3D12DescriptorHeap *pDescriptorHeaps[] = { mDescriptorHeap.Get() };
	mFallbackCommandList->SetDescriptorHeaps(ARRAYSIZE(pDescriptorHeaps), pDescriptorHeaps);

	std::vector<std::pair<ComPtr<ID3D12Resource>, uint32_t>> vertexBuffers;
	vertexBuffers.emplace_back(std::make_pair(mVertexBuffer, 3));

	AccelerationStructure blas = CreateBottomLevelAS(vertexBuffers);

	mInstances.emplace_back(std::make_pair(blas.mResult, XMMatrixIdentity()));

	UINT resultSizeInBytes = 0;
	AccelerationStructure tlas = CreateTopLevelAS(mInstances, resultSizeInBytes);
	mTlasWrappedPointer = CreateFallbackWrappedPointer(tlas.mResult.Get(), resultSizeInBytes / sizeof(UINT32));

	m_deviceResources->ExecuteCommandList();
	m_deviceResources->WaitForGpu();

	// Retain the bottom level AS result buffer and release the rest of the buffers
	mBlas = blas.mResult;

	mTlas = tlas;
}

ComPtr<ID3D12RootSignature> DXRNvTutorialApp::CreateRayGenSignature()
{
	nv_helpers_dx12::RootSignatureGenerator rootSigGenerator;
	rootSigGenerator.AddHeapRangesParameter({
		{0 /*u0*/, 1 /*1 descriptor*/, 0 /*space 0*/, D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0 /*heap slot for this UAV*/}, // output buffer
		{0 /*t0*/, 1, 0, D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1} // Tlas
	});

	return rootSigGenerator.Generate(mFallbackDevice.Get(), true);
}

ComPtr<ID3D12RootSignature> DXRNvTutorialApp::CreateMissSignature()
{
	nv_helpers_dx12::RootSignatureGenerator rootSigGenerator;
	return rootSigGenerator.Generate(mFallbackDevice.Get(), true);
}

ComPtr<ID3D12RootSignature> DXRNvTutorialApp::CreateHitSignature()
{
	nv_helpers_dx12::RootSignatureGenerator rootSigGenerator;
	return rootSigGenerator.Generate(mFallbackDevice.Get(), true);
}

ComPtr<ID3D12RootSignature> DXRNvTutorialApp::CreateGlobalRootSignature()
{
	nv_helpers_dx12::RootSignatureGenerator rootSigGenerator;
	return rootSigGenerator.Generate(mFallbackDevice.Get(), false);
}

void DXRNvTutorialApp::CreateRaytracingPipeline()
{
	auto device = m_deviceResources->GetD3DDevice();
	nv_helpers_dx12::RayTracingPipelineGenerator pipeline(device, mFallbackDevice.Get());

	// Load DXIL libraries
	pipeline.AddLibrary(g_pRaytracing, ARRAYSIZE(g_pRaytracing), {L"RayGen", L"Miss", L"ClosestHit"});

	// Create local root signatures
	mRayGenSignature = CreateRayGenSignature();
	mMissSignature = CreateMissSignature();
	mHitSignature = CreateHitSignature();

	pipeline.AddHitGroup(L"HitGroup", L"ClosestHit");

	// The following section associates the root signature to each shader. Note that we can explicitly
	// show that some shaders share the same root signature (eg. Miss and ShadowMiss). Note that the
	// hit shaders are now only referred to as hit groups, meaning that the underlying intersection,
	// any-hit and closest-hit shaders share the same root signature.
	pipeline.AddRootSignatureAssociation(mRayGenSignature.Get(), {L"RayGen"});
	pipeline.AddRootSignatureAssociation(mMissSignature.Get(), {L"Miss"});
	pipeline.AddRootSignatureAssociation(mHitSignature.Get(), {L"HitGroup"});

	// The payload size defines the maximum size of the data carried by the rays, ie. the the data
	// exchanged between shaders, such as the HitInfo structure in the HLSL code. It is important to
	// keep this value as low as possible as a too high value would result in unnecessary memory
	// consumption and cache trashing.
	pipeline.SetMaxPayloadSize(4 * sizeof(float)); // RGB + distance

	// Upon hitting a surface, DXR can provide several attributes to the hit. In our sample we just
	// use the barycentric coordinates defined by the weights u,v of the last two vertices of the
	// triangle. The actual barycentrics can be obtained using float3 barycentrics = float3(1.f-u-v,
	// u, v);
	pipeline.SetMaxAttributeSize(2 * sizeof(float)); // barycentric coordinates

	// The raytracing process can shoot rays from existing hit points, resulting in nested TraceRay
	// calls. Our sample code traces only primary rays, which then requires a trace depth of 1. Note
	// that this recursion depth should be kept to a minimum for best performance. Path tracing
	// algorithms can be easily flattened into a simple loop in the ray generation.
	pipeline.SetMaxRecursionDepth(1);

	mFallbackStateObject = pipeline.FallbackGenerate(mGlobalRootSignature.Get());
}

void DXRNvTutorialApp::CreateRaytracingOutputBuffer()
{
	auto device = m_deviceResources->GetD3DDevice();
	auto backbufferFormat = m_deviceResources->GetBackBufferFormat();

	// Create the output resource. The dimensions and format should match the swap-chain.
	auto uavDesc = CD3DX12_RESOURCE_DESC::Tex2D(backbufferFormat, m_width, m_height, 1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

	auto defaultHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
	ThrowIfFailed(device->CreateCommittedResource(
		&defaultHeapProperties, D3D12_HEAP_FLAG_NONE, &uavDesc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&mOutputResource)));
	NAME_D3D12_OBJECT(mOutputResource);

	D3D12_CPU_DESCRIPTOR_HANDLE uavDescriptorHandle;
	UINT outputResourceUAVDescriptorHeapIndex = AllocateDescriptor(&uavDescriptorHandle, UINT_MAX);
	D3D12_UNORDERED_ACCESS_VIEW_DESC UAVDesc = {};
	UAVDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
	device->CreateUnorderedAccessView(mOutputResource.Get(), nullptr, &UAVDesc, uavDescriptorHandle);
	mOutputResourceUAVGpuDescriptor = CD3DX12_GPU_DESCRIPTOR_HANDLE(mDescriptorHeap->GetGPUDescriptorHandleForHeapStart(), outputResourceUAVDescriptorHeapIndex, mDescriptorSize);
}

void DXRNvTutorialApp::InitShaderResourceHeap()
{
	auto device = m_deviceResources->GetD3DDevice();

	// Get a handle to the heap memory on the CPU side, to be able to write the descriptors directly
	D3D12_CPU_DESCRIPTOR_HANDLE srvHandle = mDescriptorHeap->GetCPUDescriptorHandleForHeapStart();

	if (false) {
		// Create the UAV. Based on the root signature we created it is the first entry. The Create*View
		// methods write the view information directly into srvHandle
		D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
		uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
		device->CreateUnorderedAccessView(mOutputResource.Get(), nullptr, &uavDesc, srvHandle);

		// Add the Top Level AS SRV right after the raytracing output buffer
		srvHandle.ptr += mDescriptorSize;
		mDescriptorsAllocated += 1;
	}

	if (false) {
		D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc;
		srvDesc.Format = DXGI_FORMAT_UNKNOWN;
		srvDesc.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
		srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		srvDesc.RaytracingAccelerationStructure.Location = mTlas.mResult->GetGPUVirtualAddress();
		// Write the acceleration structure view in the heap
		device->CreateShaderResourceView(nullptr, &srvDesc, srvHandle);

		mDescriptorsAllocated += 1;
	}
}

void DXRNvTutorialApp::CreateShaderBindingTable()
{
	auto device = m_deviceResources->GetD3DDevice();

	mShaderTableGenerator.Reset();

	// The pointer to the beginning of the heap is the only parameter required by shaders without root
	// parameters
	D3D12_GPU_DESCRIPTOR_HANDLE srvUavHeapHandle = mDescriptorHeap->GetGPUDescriptorHandleForHeapStart();
	auto heapPointer = reinterpret_cast<UINT64*>(srvUavHeapHandle.ptr);

	mShaderTableGenerator.AddRayGenerationProgram(L"RayGen", {heapPointer});
	mShaderTableGenerator.AddMissProgram(L"Miss", {});
	mShaderTableGenerator.AddHitGroup(L"HitGroup", {});

	UINT32 shaderTableSize = mShaderTableGenerator.ComputeSBTSize(mFallbackDevice.Get());

	// Create the SBT on the upload heap. This is required as the helper will use mapping to write the
	// SBT contents. After the SBT compilation it could be copied to the default heap for performance.
	mShaderTable = nv_helpers_dx12::CreateBuffer(device, shaderTableSize, D3D12_RESOURCE_FLAG_NONE,
		D3D12_RESOURCE_STATE_GENERIC_READ,
		nv_helpers_dx12::kUploadHeapProps);
	if (!mShaderTable) {
		throw std::logic_error("Could not allocate shader table");
	}

	mShaderTableGenerator.Generate(mShaderTable.Get(), mFallbackStateObject.Get());
}

void DXRNvTutorialApp::OnKeyDown(UINT8 key)
{
    switch (key) {
    case VK_SPACE:
		mRaytracingEnabled ^= true;
        break;
    default:
        break;
    }
}

// Update frame-based values.
void DXRNvTutorialApp::OnUpdate()
{
    mTimer.Tick();
    CalculateFrameStats();
    float elapsedTime = static_cast<float>(mTimer.GetElapsedSeconds());
    auto frameIndex = m_deviceResources->GetCurrentFrameIndex();
    auto prevFrameIndex = m_deviceResources->GetPreviousFrameIndex();
}

void DXRNvTutorialApp::UpdateForSizeChange(UINT width, UINT height)
{
    DXSample::UpdateForSizeChange(width, height);
}

void DXRNvTutorialApp::CreateWindowSizeDependentResources()
{
	CreateRaytracingOutputBuffer();

    // UpdateCameraMatrices();
}

void DXRNvTutorialApp::ReleaseWindowSizeDependentResources()
{
	mOutputResource.Reset();
}

void DXRNvTutorialApp::ReleaseDeviceDependentResources()
{
    mFallbackDevice.Reset();
    mFallbackCommandList.Reset();
    mFallbackStateObject.Reset();
    
	mShaderTable.Reset();

	mGlobalRootSignature.Reset();
	mRayGenSignature.Reset();
	mHitSignature.Reset();
	mMissSignature.Reset();
	mShadowSignature.Reset();
}

void DXRNvTutorialApp::DoRaytracing()
{
    auto commandList = m_deviceResources->GetCommandList();
    auto frameIndex = m_deviceResources->GetCurrentFrameIndex();

	commandList->SetComputeRootSignature(mGlobalRootSignature.Get());

	ID3D12DescriptorHeap *pDescriptorHeaps[] = { mDescriptorHeap.Get() };
	mFallbackCommandList->SetDescriptorHeaps(ARRAYSIZE(pDescriptorHeaps), pDescriptorHeaps);

	D3D12_FALLBACK_DISPATCH_RAYS_DESC desc = {};

	// The ray generation shaders are always at the beginning of the SBT. 
	uint32_t rayGenerationSectionSizeInBytes = mShaderTableGenerator.GetRayGenSectionSize();
	desc.RayGenerationShaderRecord.StartAddress = mShaderTable->GetGPUVirtualAddress();
	desc.RayGenerationShaderRecord.SizeInBytes = rayGenerationSectionSizeInBytes;

	// The miss shaders are in the second SBT section, right after the ray
	// generation shader. We have one miss shader for the camera rays and one
	// for the shadow rays, so this section has a size of 2*m_sbtEntrySize. We
	// also indicate the stride between the two miss shaders, which is the size
	// of a SBT entry
	uint32_t missSectionSizeInBytes = mShaderTableGenerator.GetMissSectionSize();
	desc.MissShaderTable.StartAddress = mShaderTable->GetGPUVirtualAddress() + rayGenerationSectionSizeInBytes;
	desc.MissShaderTable.SizeInBytes = missSectionSizeInBytes;
	desc.MissShaderTable.StrideInBytes = mShaderTableGenerator.GetMissEntrySize();

	// The hit groups section start after the miss shaders. In this sample we
	// have one 1 hit group for the triangle
	uint32_t hitGroupsSectionSize = mShaderTableGenerator.GetHitGroupSectionSize();
	desc.HitGroupTable.StartAddress = mShaderTable->GetGPUVirtualAddress() +
		rayGenerationSectionSizeInBytes +
		missSectionSizeInBytes;
	desc.HitGroupTable.SizeInBytes = hitGroupsSectionSize;
	desc.HitGroupTable.StrideInBytes = mShaderTableGenerator.GetHitGroupEntrySize();

	// Dimensions of the image to render, identical to a kernel launch dimension
	desc.Width = GetWidth();
	desc.Height = GetHeight();

	mFallbackCommandList->SetTopLevelAccelerationStructure(GlobalRootSignatureParams::AccelerationStructureSlot, mTlasWrappedPointer);

	mFallbackCommandList->DispatchRays(mFallbackStateObject.Get(), &desc);
}

void DXRNvTutorialApp::CopyRaytracingOutputToBackbuffer()
{
	auto commandList= m_deviceResources->GetCommandList();
	auto renderTarget = m_deviceResources->GetRenderTarget();

	D3D12_RESOURCE_BARRIER preCopyBarriers[2];
	preCopyBarriers[0] = CD3DX12_RESOURCE_BARRIER::Transition(renderTarget, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_DEST);
	preCopyBarriers[1] = CD3DX12_RESOURCE_BARRIER::Transition(mOutputResource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
	commandList->ResourceBarrier(ARRAYSIZE(preCopyBarriers), preCopyBarriers);

	commandList->CopyResource(renderTarget, mOutputResource.Get());

	D3D12_RESOURCE_BARRIER postCopyBarriers[2];
	postCopyBarriers[0] = CD3DX12_RESOURCE_BARRIER::Transition(renderTarget, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT);
	postCopyBarriers[1] = CD3DX12_RESOURCE_BARRIER::Transition(mOutputResource.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

	commandList->ResourceBarrier(ARRAYSIZE(postCopyBarriers), postCopyBarriers);
}

void DXRNvTutorialApp::OnRender()
{
    if (!m_deviceResources->IsWindowVisible()) {
        return;
    }

    m_deviceResources->Prepare();

	if (mRaytracingEnabled) {
//		DoRaytracing();
		DoRaytracing2();
		CopyRaytracingOutputToBackbuffer();

		m_deviceResources->Present(D3D12_RESOURCE_STATE_PRESENT);
	} else {
		auto commandList= m_deviceResources->GetCommandList();
		auto rtvHandle = m_deviceResources->GetRenderTargetView();

		const float clearColor[] = { 0.3f, 0.2f, 0.1f, 1.0f };
		commandList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);

		m_deviceResources->Present(D3D12_RESOURCE_STATE_RENDER_TARGET);
	}
}

void DXRNvTutorialApp::OnDestroy()
{
    m_deviceResources->WaitForGpu();
    OnDeviceLost();
}

void DXRNvTutorialApp::OnDeviceLost()
{
    ReleaseWindowSizeDependentResources();
    ReleaseDeviceDependentResources();
}

void DXRNvTutorialApp::OnDeviceRestored()
{
    CreateDeviceDependentResources();
    CreateWindowSizeDependentResources();
}

void DXRNvTutorialApp::CalculateFrameStats()
{
    static int frameCnt = 0;
    static double elapsedTime = 0.0f;
    double totalTime = mTimer.GetTotalSeconds();
    frameCnt++;

    // Compute averages over one second period.
    if ((totalTime - elapsedTime) >= 1.0f) {
        float diff = static_cast<float>(totalTime - elapsedTime);
        float fps = static_cast<float>(frameCnt) / diff; // Normalize to an exact second.

        frameCnt = 0;
        elapsedTime = totalTime;

        float MRaysPerSecond = (m_width * m_height * fps) / static_cast<float>(1e6);

        wstringstream windowText;

		if (mFallbackDevice->UsingRaytracingDriver()) {
			windowText << L"(FL-DXR)";
		} else {
			windowText << L"(FL)";
		}
        windowText << setprecision(2) << fixed
            << L"    fps: " << fps << L"     ~Million Primary Rays/s: " << MRaysPerSecond
            << L"    GPU[" << m_deviceResources->GetAdapterID() << L"]: " << m_deviceResources->GetAdapterDescription();
        SetCustomWindowText(windowText.str().c_str());
    }
}

void DXRNvTutorialApp::OnSizeChanged(UINT width, UINT height, bool minimized)
{
    if (!m_deviceResources->WindowSizeChanged(width, height, minimized)) {
        return;
    }

    UpdateForSizeChange(width, height);

    ReleaseWindowSizeDependentResources();
    CreateWindowSizeDependentResources();
}

// Create a wrapped pointer for the Fallback Layer path.
WRAPPED_GPU_POINTER DXRNvTutorialApp::CreateFallbackWrappedPointer(ID3D12Resource* resource, UINT bufferNumElements)
{
    auto device = m_deviceResources->GetD3DDevice();

    D3D12_UNORDERED_ACCESS_VIEW_DESC rawBufferUavDesc = {};
    rawBufferUavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    rawBufferUavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    rawBufferUavDesc.Format = DXGI_FORMAT_R32_TYPELESS;
    rawBufferUavDesc.Buffer.NumElements = bufferNumElements;

    D3D12_CPU_DESCRIPTOR_HANDLE bottomLevelDescriptor;
   
    // Only compute fallback requires a valid descriptor index when creating a wrapped pointer.
    UINT descriptorHeapIndex = 0;
    if (!mFallbackDevice->UsingRaytracingDriver()) {
        descriptorHeapIndex = AllocateDescriptor(&bottomLevelDescriptor);
        device->CreateUnorderedAccessView(resource, nullptr, &rawBufferUavDesc, bottomLevelDescriptor);
    }
    return mFallbackDevice->GetWrappedPointerSimple(descriptorHeapIndex, resource->GetGPUVirtualAddress());
}

// Allocate a descriptor and return its index. 
// If the passed descriptorIndexToUse is valid, it will be used instead of allocating a new one.
UINT DXRNvTutorialApp::AllocateDescriptor(D3D12_CPU_DESCRIPTOR_HANDLE* cpuDescriptor, UINT descriptorIndexToUse)
{
    auto descriptorHeapCpuBase = mDescriptorHeap->GetCPUDescriptorHandleForHeapStart();
    if (descriptorIndexToUse >= mDescriptorHeap->GetDesc().NumDescriptors) {
        descriptorIndexToUse = mDescriptorsAllocated++;
    }
    *cpuDescriptor = CD3DX12_CPU_DESCRIPTOR_HANDLE(descriptorHeapCpuBase, descriptorIndexToUse, mDescriptorSize);
    return descriptorIndexToUse;
}


////////////////////////////////////////////////////////////////////////////////

// Local root signature and shader association
// This is a root signature that enables a shader to have unique arguments that come from shader tables.
void DXRNvTutorialApp::CreateLocalRootSignatureSubobjects(CD3D12_STATE_OBJECT_DESC* raytracingPipeline)
{
	// Local root signature to be used in a hit group.
	auto localRootSignature = raytracingPipeline->CreateSubobject<CD3D12_LOCAL_ROOT_SIGNATURE_SUBOBJECT>();
	localRootSignature->SetRootSignature(mRayGenSignature.Get());
	// Define explicit shader association for the local root signature. 
	// In this sample, this could be ommited for convenience since it matches the default association.
	{
		auto rootSignatureAssociation = raytracingPipeline->CreateSubobject<CD3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION_SUBOBJECT>();
		rootSignatureAssociation->SetSubobjectToAssociate(*localRootSignature);
		rootSignatureAssociation->AddExport(L"RayGen");
	}

	// Empty local root signature to be used in a ray gen and a miss shader.
	{
		auto localRootSignature = raytracingPipeline->CreateSubobject<CD3D12_LOCAL_ROOT_SIGNATURE_SUBOBJECT>();
		localRootSignature->SetRootSignature(mMissSignature.Get());
		// Shader association
		auto rootSignatureAssociation = raytracingPipeline->CreateSubobject<CD3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION_SUBOBJECT>();
		rootSignatureAssociation->SetSubobjectToAssociate(*localRootSignature);
		rootSignatureAssociation->AddExport(L"Miss");
		rootSignatureAssociation->AddExport(L"HitGroup");
	}
}

// Create a raytracing pipeline state object (RTPSO).
// An RTPSO represents a full set of shaders reachable by a DispatchRays() call,
// with all configuration options resolved, such as local signatures and other state.
void DXRNvTutorialApp::CreateRaytracingPipelineStateObject()
{
	mRayGenSignature = CreateRayGenSignature();
	mMissSignature = CreateMissSignature();
	mHitSignature = CreateHitSignature();

	// Create 7 subobjects that combine into a RTPSO:
	// Subobjects need to be associated with DXIL exports (i.e. shaders) either by way of default or explicit associations.
	// Default association applies to every exported shader entrypoint that doesn't have any of the same type of subobject associated with it.
	// This simple sample utilizes default shader association except for local root signature subobject
	// which has an explicit association specified purely for demonstration purposes.
	// 1 - DXIL library
	// 1 - Triangle hit group
	// 1 - Shader config
	// 2 - Local root signature and association
	// 1 - Global root signature
	// 1 - Pipeline config
	CD3D12_STATE_OBJECT_DESC raytracingPipeline{ D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE };


	// DXIL library
	// This contains the shaders and their entrypoints for the state object.
	// Since shaders are not considered a subobject, they need to be passed in via DXIL library subobjects.
	auto lib = raytracingPipeline.CreateSubobject<CD3D12_DXIL_LIBRARY_SUBOBJECT>();
	D3D12_SHADER_BYTECODE libdxil = CD3DX12_SHADER_BYTECODE((void *)g_pRaytracing, ARRAYSIZE(g_pRaytracing));
	lib->SetDXILLibrary(&libdxil);
	// Define which shader exports to surface from the library.
	// If no shader exports are defined for a DXIL library subobject, all shaders will be surfaced.
	// In this sample, this could be ommited for convenience since the sample uses all shaders in the library. 
	{
		lib->DefineExport(L"RayGen");
		lib->DefineExport(L"ClosestHit");
		lib->DefineExport(L"Miss");
	}

	// Triangle hit group
	// A hit group specifies closest hit, any hit and intersection shaders to be executed when a ray intersects the geometry's triangle/AABB.
	// In this sample, we only use triangle geometry with a closest hit shader, so others are not set.
	auto hitGroup = raytracingPipeline.CreateSubobject<CD3D12_HIT_GROUP_SUBOBJECT>();
	hitGroup->SetClosestHitShaderImport(L"ClosestHit");
	hitGroup->SetHitGroupExport(L"HitGroup");

	// Shader config
	// Defines the maximum sizes in bytes for the ray payload and attribute structure.
	auto shaderConfig = raytracingPipeline.CreateSubobject<CD3D12_RAYTRACING_SHADER_CONFIG_SUBOBJECT>();
	UINT payloadSize = sizeof(XMFLOAT4);    // float4 pixelColor
	UINT attributeSize = sizeof(XMFLOAT2);  // float2 barycentrics
	shaderConfig->Config(payloadSize, attributeSize);

	// Local root signature and shader association
	// This is a root signature that enables a shader to have unique arguments that come from shader tables.
	CreateLocalRootSignatureSubobjects(&raytracingPipeline);

	// Global root signature
	// This is a root signature that is shared across all raytracing shaders invoked during a DispatchRays() call.
	auto globalRootSignature = raytracingPipeline.CreateSubobject<CD3D12_ROOT_SIGNATURE_SUBOBJECT>();
	globalRootSignature->SetRootSignature(mGlobalRootSignature.Get());

	// Pipeline config
	// Defines the maximum TraceRay() recursion depth.
	auto pipelineConfig = raytracingPipeline.CreateSubobject<CD3D12_RAYTRACING_PIPELINE_CONFIG_SUBOBJECT>();
	// PERFOMANCE TIP: Set max recursion depth as low as needed 
	// as drivers may apply optimization strategies for low recursion depths.
	UINT maxRecursionDepth = 1; // ~ primary rays only. 
	pipelineConfig->Config(maxRecursionDepth);

#if _DEBUG
	PrintStateObjectDesc(raytracingPipeline);
#endif

	// Create the state object.
	ThrowIfFailed(mFallbackDevice->CreateStateObject(raytracingPipeline, IID_PPV_ARGS(&mFallbackStateObject)), L"Couldn't create DirectX Raytracing state object.\n");
}

void DXRNvTutorialApp::BuildShaderTables()
{
	auto device = m_deviceResources->GetD3DDevice();

	void* rayGenShaderIdentifier;
	void* missShaderIdentifier;
	void* hitGroupShaderIdentifier;

	auto GetShaderIdentifiers = [&](auto* stateObjectProperties)
	{
		rayGenShaderIdentifier = stateObjectProperties->GetShaderIdentifier(L"RayGen");
		missShaderIdentifier = stateObjectProperties->GetShaderIdentifier(L"Miss");
		hitGroupShaderIdentifier = stateObjectProperties->GetShaderIdentifier(L"HitGroup");
	};

	// Get shader identifiers.
	UINT shaderIdentifierSize;
	GetShaderIdentifiers(mFallbackStateObject.Get());
	shaderIdentifierSize = mFallbackDevice->GetShaderIdentifierSize();

	// Ray gen shader table
	{
		UINT numShaderRecords = 1;
		UINT shaderRecordSize = shaderIdentifierSize;
		ShaderTable rayGenShaderTable(device, numShaderRecords, shaderRecordSize, L"RayGenShaderTable");
		rayGenShaderTable.push_back(ShaderRecord(rayGenShaderIdentifier, shaderIdentifierSize));
		m_rayGenShaderTable = rayGenShaderTable.GetResource();
	}

	// Miss shader table
	{
		UINT numShaderRecords = 1;
		UINT shaderRecordSize = shaderIdentifierSize;
		ShaderTable missShaderTable(device, numShaderRecords, shaderRecordSize, L"MissShaderTable");
		missShaderTable.push_back(ShaderRecord(missShaderIdentifier, shaderIdentifierSize));
		m_missShaderTable = missShaderTable.GetResource();
	}

	// Hit group shader table
	{
		UINT numShaderRecords = 1;
		UINT shaderRecordSize = shaderIdentifierSize;
		ShaderTable hitGroupShaderTable(device, numShaderRecords, shaderRecordSize, L"HitGroupShaderTable");
		hitGroupShaderTable.push_back(ShaderRecord(hitGroupShaderIdentifier, shaderIdentifierSize));
		m_hitGroupShaderTable = hitGroupShaderTable.GetResource();
	}
}

void DXRNvTutorialApp::DoRaytracing2()
{
	auto commandList = m_deviceResources->GetCommandList();
	auto frameIndex = m_deviceResources->GetCurrentFrameIndex();

	auto DispatchRays = [&](auto* commandList, auto* stateObject, auto* dispatchDesc)
	{
		// Since each shader table has only one shader record, the stride is same as the size.
		dispatchDesc->HitGroupTable.StartAddress = m_hitGroupShaderTable->GetGPUVirtualAddress();
		dispatchDesc->HitGroupTable.SizeInBytes = m_hitGroupShaderTable->GetDesc().Width;
		dispatchDesc->HitGroupTable.StrideInBytes = dispatchDesc->HitGroupTable.SizeInBytes;
		dispatchDesc->MissShaderTable.StartAddress = m_missShaderTable->GetGPUVirtualAddress();
		dispatchDesc->MissShaderTable.SizeInBytes = m_missShaderTable->GetDesc().Width;
		dispatchDesc->MissShaderTable.StrideInBytes = dispatchDesc->MissShaderTable.SizeInBytes;
		dispatchDesc->RayGenerationShaderRecord.StartAddress = m_rayGenShaderTable->GetGPUVirtualAddress();
		dispatchDesc->RayGenerationShaderRecord.SizeInBytes = m_rayGenShaderTable->GetDesc().Width;
		dispatchDesc->Width = m_width;
		dispatchDesc->Height = m_height;
		commandList->DispatchRays(stateObject, dispatchDesc);
	};

	auto SetCommonPipelineState = [&](auto* descriptorSetCommandList)
	{
		descriptorSetCommandList->SetDescriptorHeaps(1, mDescriptorHeap.GetAddressOf());
		// Set index and successive vertex buffer decriptor tables
//		commandList->SetComputeRootDescriptorTable(GlobalRootSignatureParams::OutputViewSlot, mOutputResourceUAVGpuDescriptor);
	};

	commandList->SetComputeRootSignature(mGlobalRootSignature.Get());

	// Bind the heaps, acceleration structure and dispatch rays.    
	D3D12_FALLBACK_DISPATCH_RAYS_DESC dispatchDesc = {};
	SetCommonPipelineState(mFallbackCommandList.Get());
	mFallbackCommandList->SetTopLevelAccelerationStructure(GlobalRootSignatureParams::AccelerationStructureSlot, mTlasWrappedPointer);
	DispatchRays(mFallbackCommandList.Get(), mFallbackStateObject.Get(), &dispatchDesc);
}

void DXRNvTutorialApp::SerializeAndCreateRaytracingRootSignature(D3D12_ROOT_SIGNATURE_DESC& desc, ComPtr<ID3D12RootSignature>* rootSig)
{
	auto device = m_deviceResources->GetD3DDevice();
	ComPtr<ID3DBlob> blob;
	ComPtr<ID3DBlob> error;

	ThrowIfFailed(mFallbackDevice->D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error), error ? static_cast<wchar_t*>(error->GetBufferPointer()) : nullptr);
	ThrowIfFailed(mFallbackDevice->CreateRootSignature(1, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&(*rootSig))));
}

void DXRNvTutorialApp::CreateRootSignatures()
{
	auto device = m_deviceResources->GetD3DDevice();

	// Global Root Signature
	// This is a root signature that is shared across all raytracing shaders invoked during a DispatchRays() call.
	{
		CD3DX12_DESCRIPTOR_RANGE ranges[2]; // Perfomance TIP: Order from most frequent to least frequent.
		ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);  // 1 output texture
		ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1);  // 1 static vertex buffer.

		CD3DX12_ROOT_PARAMETER rootParameters[GlobalRootSignatureParams::Count];
		rootParameters[GlobalRootSignatureParams::OutputViewSlot].InitAsDescriptorTable(1, &ranges[0]);
		rootParameters[GlobalRootSignatureParams::AccelerationStructureSlot].InitAsShaderResourceView(0);
		CD3DX12_ROOT_SIGNATURE_DESC globalRootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters);
		SerializeAndCreateRaytracingRootSignature(globalRootSignatureDesc, &mGlobalRootSignature);
	}

#if 0
	// Local Root Signature
	// This is a root signature that enables a shader to have unique arguments that come from shader tables.
	{
		CD3DX12_ROOT_PARAMETER rootParameters[LocalRootSignatureParams::Count];
		rootParameters[LocalRootSignatureParams::CubeConstantSlot].InitAsConstants(SizeOfInUint32(m_cubeCB), 1);
		CD3DX12_ROOT_SIGNATURE_DESC localRootSignatureDesc(ARRAYSIZE(rootParameters), rootParameters);
		localRootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE;
		SerializeAndCreateRaytracingRootSignature(localRootSignatureDesc, &m_raytracingLocalRootSignature);
	}
	// Empty local root signature
	{
		CD3DX12_ROOT_SIGNATURE_DESC localRootSignatureDesc(D3D12_DEFAULT);
		localRootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE;
		SerializeAndCreateRaytracingRootSignature(localRootSignatureDesc, &m_raytracingLocalRootSignatureEmpty);
	}
#endif
}
