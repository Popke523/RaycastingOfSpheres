#pragma once

#include "cuda_runtime.h"

struct lightSource
{
	float3 position;
	float3 color;
	float intensity;
};