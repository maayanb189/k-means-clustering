#include "Kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>


#ifdef PARALLEL
int getNumOfBlock(int pointsArrSize, cudaDeviceProp prop);
void checkErrorStatus(cudaError e);

__device__ double calcDistanceCuda(double location1[DIMENSIONS], double location2[DIMENSIONS])
{
	double powDistanceSum = 0;
	int dimensionIndex;
	for (dimensionIndex = 0; dimensionIndex < DIMENSIONS; dimensionIndex++)
	{
		powDistanceSum += pow((location1[dimensionIndex] - location2[dimensionIndex]), 2);
	}
	double sqrtD = sqrt(powDistanceSum);
	return sqrtD;
}

__global__ void updatePointsLocation(Point * points, int numOfPoints, double timeInterval, int numOfThreadsPerBlock)
{
	int pointIndex = blockIdx.x*numOfThreadsPerBlock + threadIdx.x;
	if (pointIndex < numOfPoints)
	{
		int axisIndex;
		for (axisIndex = 0; axisIndex < DIMENSIONS; axisIndex++)
		{
			points[pointIndex].location[axisIndex] += timeInterval*points[pointIndex].velocity[axisIndex];
		}
	}
}

__global__ void groupPoints(Cluster * clusters, int numOfClusters, Point ** points, int numOfPoints, int numOfThreadsPerBlock, bool * pointsMoved)
{
	//*pointsMoved = false;
	int pointIndex = blockIdx.x*numOfThreadsPerBlock + threadIdx.x;

	if (pointIndex < numOfPoints)
	{
		Point * currentPoint = &((*points)[pointIndex]);

		int clusterIndex = 0;
		double minDistance = calcDistanceCuda(currentPoint->location, clusters[clusterIndex].center);
		int closestClusterIndex = clusterIndex;
		for (clusterIndex = 1; clusterIndex < numOfClusters; clusterIndex++)
		{
			double distance = calcDistanceCuda(currentPoint->location, clusters[clusterIndex].center);
			if (distance < minDistance)
			{
				minDistance = distance;
				closestClusterIndex = clusterIndex;
			}
		}
		//update the current cluster
		if (currentPoint->currentCluster != closestClusterIndex)
		{
			currentPoint->currentCluster = closestClusterIndex;
			*pointsMoved = true;
		}
	}
}


Point * allocatePointsOnGpuCuda(Point * points, int numOfPoints)
{
	Point * pointsGPU;
	cudaError cudaStatus;
	
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	checkErrorStatus(cudaStatus);

	// Allocate GPU buffers for points
	cudaStatus = cudaMalloc((void**)&pointsGPU, numOfPoints * sizeof(Point));
	checkErrorStatus(cudaStatus);

	//cuda memcpy to GPU
	cudaStatus = cudaMemcpy(pointsGPU, points, numOfPoints * sizeof(Point), cudaMemcpyHostToDevice);
	checkErrorStatus(cudaStatus);

	return pointsGPU;
}

Cluster * allocateClustersOnGPU(Cluster * clusters)
{
	Cluster * clustersGPU;
	cudaError cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	checkErrorStatus(cudaStatus);

	// Allocate GPU buffers for clusters
	cudaStatus = cudaMalloc((void**)&clustersGPU, params.numOfClusters * sizeof(Cluster));
	checkErrorStatus(cudaStatus);

	//cuda memcpy to GPU
	cudaStatus = cudaMemcpy(clustersGPU, clusters, params.numOfClusters * sizeof(Cluster), cudaMemcpyHostToDevice);
	checkErrorStatus(cudaStatus);

	return clustersGPU;
}

Point * progressPointsLocationCuda(Point * points, int numOfPoints, Point * pointArr_onGPU)
{
	cudaError_t cudaStatus;
	cudaDeviceProp prop;
	int numOfBlocks;
	//int pointIndex;

	//get device properties
	cudaStatus = cudaGetDeviceProperties(&prop, 0);
	checkErrorStatus(cudaStatus);

	numOfBlocks = getNumOfBlock(numOfPoints, prop);

	// Choose which GPU to run on
	cudaStatus = cudaSetDevice(0);
	checkErrorStatus(cudaStatus);

	updatePointsLocation << <numOfBlocks, prop.maxThreadsPerBlock >> >(pointArr_onGPU, numOfPoints, params.timeInterval, prop.maxThreadsPerBlock);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	checkErrorStatus(cudaStatus);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	checkErrorStatus(cudaStatus);

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(points, pointArr_onGPU, numOfPoints * sizeof(Point), cudaMemcpyDeviceToHost);
	checkErrorStatus(cudaStatus);

//	for (pointIndex = 0; pointIndex < numOfPoints; pointIndex++)
//	{
//		cudaStatus = cudaMemcpy(points->location, pointArr_onGPU->location, sizeof(points->location), cudaMemcpyDeviceToHost);
//		checkErrorStatus(cudaStatus);
//	}
	return pointArr_onGPU;
}

//returns true if points have moved, else - false
bool groupPointsCuda(Cluster * clusters, Point ** pointsOnGPU, int numOfPoints)
{
	bool ret = false;
	cudaError_t cudaStatus;
	cudaDeviceProp prop;

	int numOfBlocks = 0;
	Cluster* clustersOnGPU;

	//get device properties
	cudaStatus = cudaGetDeviceProperties(&prop, 0);
	checkErrorStatus(cudaStatus);

	cudaStatus = cudaSetDevice(0);
	checkErrorStatus(cudaStatus);

	clustersOnGPU = allocateClustersOnGPU(clusters);

	numOfBlocks = getNumOfBlock(numOfPoints, prop);
	printf("numOfthreads : %d , numOfBlocks : %d\n", prop.maxThreadsPerBlock , numOfBlocks);
	fflush(stdout);

	
	ret = false;
	groupPoints << <numOfBlocks*2, prop.maxThreadsPerBlock/2 >> > (clustersOnGPU, params.numOfClusters, pointsOnGPU, numOfPoints, prop.maxThreadsPerBlock/2, &ret);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	checkErrorStatus(cudaStatus);

	
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	checkErrorStatus(cudaStatus);

	
	//free cuda clusters
	cudaFree(clustersOnGPU);

	
	return ret;
}


int getNumOfBlock(int numOfPoints, cudaDeviceProp prop)
{
	int numOfBlocks = numOfPoints / prop.maxThreadsPerBlock;
	if (numOfPoints % prop.maxThreadsPerBlock)
	{
		numOfBlocks++;
	}
	return numOfBlocks;
}

void checkErrorStatus(cudaError e)
{
	// check if the status from the cuda was ok
	if (e != cudaSuccess)
	{
		printf("Cuda Error: %d\n", e);
		fflush(stdout);
		exit(1);
	}
}

#endif

