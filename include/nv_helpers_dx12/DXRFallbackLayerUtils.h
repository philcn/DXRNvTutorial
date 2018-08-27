#pragma once
#include "D3D12RaytracingFallback.h"
#include "d3dx12.h"

// Allocate a descriptor and return its index. 
// If the passed descriptorIndexToUse is valid, it will be used instead of allocating a new one.
inline UINT AllocateDescriptor(ID3D12DescriptorHeap *descriptorHeap, UINT &descriptorsAllocated, UINT descriptorSize, 
	D3D12_CPU_DESCRIPTOR_HANDLE* cpuDescriptor, UINT descriptorIndexToUse = UINT_MAX)
{
	auto descriptorHeapCpuBase = descriptorHeap->GetCPUDescriptorHandleForHeapStart();
	if (descriptorIndexToUse >= descriptorHeap->GetDesc().NumDescriptors) {
		descriptorIndexToUse = descriptorsAllocated++;
	}
	*cpuDescriptor = CD3DX12_CPU_DESCRIPTOR_HANDLE(descriptorHeapCpuBase, descriptorIndexToUse, descriptorSize);
	return descriptorIndexToUse;
}

// Create a wrapped pointer for the Fallback Layer path.
inline WRAPPED_GPU_POINTER CreateFallbackWrappedPointer(
	ID3D12Device *device, ID3D12RaytracingFallbackDevice *fallbackDevice, ID3D12DescriptorHeap *descriptorHeap, 
	UINT &descriptorsAllocated, UINT descriptorSize, ID3D12Resource *resource, UINT bufferNumElements)
{
	D3D12_UNORDERED_ACCESS_VIEW_DESC rawBufferUavDesc = {};
	rawBufferUavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
	rawBufferUavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
	rawBufferUavDesc.Format = DXGI_FORMAT_R32_TYPELESS;
	rawBufferUavDesc.Buffer.NumElements = bufferNumElements;

	D3D12_CPU_DESCRIPTOR_HANDLE bottomLevelDescriptor;

	// Only compute fallback requires a valid descriptor index when creating a wrapped pointer.
	UINT descriptorHeapIndex = 0;
	if (!fallbackDevice->UsingRaytracingDriver())
	{
		descriptorHeapIndex = AllocateDescriptor(descriptorHeap, descriptorsAllocated, descriptorSize, &bottomLevelDescriptor);
		device->CreateUnorderedAccessView(resource, nullptr, &rawBufferUavDesc, bottomLevelDescriptor);
	}
	return fallbackDevice->GetWrappedPointerSimple(descriptorHeapIndex, resource->GetGPUVirtualAddress());
}
