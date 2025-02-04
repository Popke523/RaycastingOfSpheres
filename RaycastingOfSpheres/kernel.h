#pragma once

#define _USE_MATH_DEFINES

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CHECK_CUDA_ERR(x)                                         \
    do {                                                          \
        cudaError_t err = x;                                      \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr,                                       \
                    "CUDA Error at %s:%d -> %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

#include <cuda_runtime.h>
#include <surface_functions.h>
#include "device_launch_parameters.h"

#include <cmath>
#include "sphere.h"
#include "lightSource.h"
#include "camera.h"

template<typename T> struct vec4
{
	T x, y, z, w;
};

__global__ void renderTestKernel(cudaSurfaceObject_t surface, int renderWidth, int renderHeight);

void renderTestKernelLauncher(cudaSurfaceObject_t surface, int renderWidth, int renderHeight, camera camera, sphere * spheres, int n_spheres, lightSource * lightSources, int n_lightSources, float brightness, float kd, float ks);

__global__ void renderKernel(cudaSurfaceObject_t surface, sphere *spheres, int spheresLength, lightSource *lightSources, int lightSourcesLength, int renderWidth, int renderHeight, camera camera, float brightness, float kd, float ks);