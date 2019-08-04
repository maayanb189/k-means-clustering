#pragma once
#include "Typedefs.h"

Point * allocatePointsOnGpuCuda(Point * points, int numOfPoints);
Point * progressPointsLocationCuda(Point * points, int numOfPoints, Point * pointArr_onGPU);
bool groupPointsCuda(Cluster * clusters, Point ** pointsOnGPU, int numOfPoints);