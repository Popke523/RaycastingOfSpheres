#pragma once

#include "sphere.h"
#include "lightSource.h"

sphere random_sphere(float min_pos, float max_pos, float min_radius, float max_radius);

lightSource random_light_source(float min_pos, float max_pos);

float3 random_float3(float min, float max);

float random_float(float min, float max);