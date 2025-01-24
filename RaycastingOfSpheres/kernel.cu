#include "kernel.h"

#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <surface_functions.h>

__global__ void renderTestKernel(cudaSurfaceObject_t surface, int renderWidth, int renderHeight)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= renderWidth * renderHeight) return;

	int x = i % renderWidth;
	int y = i / renderWidth;

	vec4<uint8_t> rgba = { (uint8_t)(x % 255), (uint8_t)(y % 255), (uint8_t)(x % 255), (uint8_t)(y % 255) };

	uint32_t bytes = rgba.w << 24 | rgba.z << 16 | rgba.y << 8 | rgba.x;

	surf2Dwrite(bytes, surface, x * sizeof(uint32_t), y);
}

void renderTestKernelLauncher(cudaSurfaceObject_t surface, int renderWidth, int renderHeight, camera camera, sphere *spheres, int n_spheres, lightSource *lightSources, int n_lightSources, float brightness, float kd, float ks)
{
	int number_of_pixels = renderWidth * renderHeight;

	int THREADS_PER_BLOCK = number_of_pixels > 1024 ? 1024 : number_of_pixels;
	int NUMBER_OF_BLOCKS = (number_of_pixels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	renderKernel << <NUMBER_OF_BLOCKS, THREADS_PER_BLOCK >> > (surface, spheres, n_spheres, lightSources, n_lightSources, renderWidth, renderHeight, camera, brightness, kd, ks);

	CHECK_CUDA_ERR(cudaDeviceSynchronize());
}



__global__ void renderKernel(cudaSurfaceObject_t surface, sphere *spheres, int spheresLength, lightSource *lightSources, int lightSourcesLength, int renderWidth, int renderHeight, camera camera, float brightness, float kd, float ks)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= renderWidth * renderHeight) return;

	int x = i % renderWidth;
	int y = i / renderWidth;

	float aspectRatio = float(renderWidth) / float(renderHeight);

	float Px = (2 * ((float(x) + 0.5) / float(renderWidth)) - 1) * tan(camera.fov_degrees * 0.5f * M_PI / 180.0f) * aspectRatio;
	float Py = (1 - (2 * ((float(y) + 0.5) / float(renderHeight)))) * tan(camera.fov_degrees * 0.5f * M_PI / 180.0f);

	float3 rayOrigin = camera.position;

	float3 rayDirection = normalize(camera_to_world_rotate(camera, make_float3(Px, Py, -1.0f)));

	float d1, d2;

	float ka = 0.01f;

	float nearest_intersection_distance = 1000000.0f;

	float3 color = make_float3(0.0f, 0.0f, 0.0f);

	int nearestSphereIndex = -1;

	for (int i = 0; i < spheresLength; i++)
	{
		if (LineIntersect(spheres[i], rayOrigin, rayDirection, d1, d2))
		{
			if (d1 < 0.0f) continue;
			if (d1 >= nearest_intersection_distance) continue;
			nearest_intersection_distance = d1;
			nearestSphereIndex = i;
		}
	}

	if (nearestSphereIndex != -1)
	{
		int i = nearestSphereIndex;
		float3 intersectionPoint = rayOrigin + nearest_intersection_distance * rayDirection;
		float3 normal = normalize(intersectionPoint - spheres[i].position);
		color = make_float3(0.0f, 0.0f, 0.0f);
		for (int j = 0; j < lightSourcesLength; j++)
		{
			float3 lightDirection = normalize(lightSources[j].position - intersectionPoint);
			float3 reflectionDirection = reflect(-lightDirection, normal);
			float3 viewDirection = normalize(rayOrigin - intersectionPoint);
			float diffuse = max(dot(lightDirection, normal), 0.0f);
			float specular = pow(max(dot(reflectionDirection, viewDirection), 0.0f), spheres[i].alpha);
			color += lightSources[j].intensity * lightSources[j].color * (spheres[i].kd * diffuse * kd + spheres[i].ks * specular * ks) * spheres[i].color;
		}

		color += spheres[i].ka * spheres[i].color;
	}

	color = brightness * color;

	color.x = min(color.x, 1.0f);
	color.y = min(color.y, 1.0f);
	color.z = min(color.z, 1.0f);

	vec4<uint8_t> rgba = { (uint8_t)(color.x * 255.0f), (uint8_t)(color.y * 255.0f), (uint8_t)(color.z * 255.0f), 255 };

	uint32_t bytes = rgba.w << 24 | rgba.z << 16 | rgba.y << 8 | rgba.x;

	surf2Dwrite(bytes, surface, x * sizeof(uint32_t), y);
}