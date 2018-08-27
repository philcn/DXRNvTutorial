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

#ifndef RAYTRACING_HLSL
#define RAYTRACING_HLSL

#define HLSL
#include "RaytracingHlslCompat.h"

// Raytracing output texture, accessed as a UAV
RWTexture2D< float4 > gOutput : register(u0);

// Raytracing acceleration structure, accessed as a SRV
RaytracingAccelerationStructure SceneBVH : register(t0);

[shader("raygeneration")] 
void RayGen() {
  // Initialize the ray payload
  HitInfo payload;
  payload.colorAndDistance = float4(0.9, 0.6, 0.2, 1);

  // Get the location within the dispatched 2D grid of work items
  // (often maps to pixels, so this could represent a pixel coordinate).
  uint2 launchIndex = DispatchRaysIndex();

  gOutput[launchIndex] = float4(payload.colorAndDistance.rgb, 1.f);
}

[shader("closesthit")] 
void ClosestHit(inout HitInfo payload, Attributes attrib) 
{
  payload.colorAndDistance = float4(1, 1, 0, RayTCurrent());
}

[shader("miss")]
void Miss(inout HitInfo payload : SV_RayPayload)
{
  payload.colorAndDistance = float4(0.2f, 0.2f, 0.8f, -1.f);
}

#endif // RAYTRACING_HLSL