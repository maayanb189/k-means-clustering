#include <stdio.h>
#include "FileHandler.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mpi.h"
#include <omp.h>
#include "MpiUtils.h"
#include "Kernel.h"


void initClusters(Cluster ** clusters, Point * points);
double calcDistance(double location1[DIMENSIONS], double location2[DIMENSIONS]);
bool groupPointsHendler(Cluster * clusters, Point ** myPointsOnCpu, Point ** pointsOnGpu, int numOfPoints);
void calcClustersCenter(Cluster ** clusters, Point * points);
void calcSingleDiameter(Cluster * clusterPtr, Point * points);
void calcClustersDiameter(Cluster ** clusters, Point * points);
bool isPartOfCluster(Cluster cluster, Point point);
double calcQuality(Cluster ** clusters, Point * points);
void progressPointsLocation(Point ** points);
int getNumOfPointsPerProc(int rankId, Point * points, int numOfProcs);
void allocatePointsArr(Point ** points, int numOfPointsPerProc);
void determineAlgoTerminate(int rankId, int numOfProcs, int * algoTermination);
void determineSuccessBasedOnQuality(int rankId, double * quality, Cluster ** clusters, Point * points, int * success);
bool groupPointsSerial(Cluster * clusters, Point ** points, int numOfPoints);

AlgoParams params;
int rankId;
int timeIndex = 0;

int main(int argc, char* argv[])
{
	printf("Start\n");
	Cluster * clusters;
	Point * myPoints;
	Point * myPointsOnGPU;
	Point * allPoints;
	int myNumOfPoints = 0;
	int success = FALSE;
	double quality;
	double timeStart, timeFinish;

	int numOfProcs;
	
	MpiUtils_initProgram(&argc, &argv, &rankId, &numOfProcs);
	
	MpiUtils_createPointType();
	MpiUtils_createClusterType();
	MpiUtils_createAlgoParamsType();
	
	if ( MASTER_ID == rankId )
	{
		char * inputFileName = "D:\\Kmeans_Maayan - EOD 20-11-18\\Kmeans_Maayan\\data.txt";
		timeStart = MPI_Wtime();
		ReadFromFile(inputFileName, &allPoints);
	}
	//Share algorithm parameters with all the processes
	MpiUtils_BcastAlgoParams();
	MpiUtils_initSendResvParams(numOfProcs);
	if( MASTER_ID == rankId )
	{
		//choose first K points as centers of clusters
		initClusters(&clusters, allPoints);
		//ClustersToFile("D:\\Kmeans_Maayan\\Kmeans_Maayan\\clusters - P.txt", clusters, timeIndex);
	}
	else
	{
		clusters = (Cluster *)malloc(params.numOfClusters * sizeof(Cluster));
	}
	//each process allocates a memory for his own points and receive them from the master  
	myNumOfPoints = getNumOfPointsPerProc(rankId, allPoints, numOfProcs);
	allocatePointsArr(&myPoints, myNumOfPoints);
	
	distributePoints(rankId, allPoints, myPoints, numOfProcs);
	myPointsOnGPU = allocatePointsOnGpuCuda(myPoints, myNumOfPoints);
	
	while ((timeIndex * params.timeInterval) <= params.endOfTime && !success)
	{
		int algoTermination = FALSE; // FALSE - points have moved, TRUE - else
		int kMeansIteration = 0;

		//Calculate Points locations using CUDA
		if(timeIndex > 0)
		{
			myPointsOnGPU = progressPointsLocationCuda(myPoints, myNumOfPoints, myPointsOnGPU);
		}

		for (kMeansIteration = 0; kMeansIteration < params.maxIterations && !algoTermination; kMeansIteration++)
		{
			MpiUtils_BcastClusters(clusters);
			algoTermination = groupPointsSerial(clusters, &myPoints, myNumOfPoints);
			collectPoints(rankId, allPoints, myPoints, numOfProcs);
			determineAlgoTerminate(rankId, numOfProcs, &algoTermination);
			if ( MASTER_ID == rankId )
			{
				calcClustersCenter(&clusters, allPoints);
			}
		}
		if (MASTER_ID == rankId)
		{
			fflush(stdout);
			printf("RankId: %d, Time: %d, kMeansIteration: %d\n", rankId, timeIndex, kMeansIteration);
			fflush(stdout);
		}
		determineSuccessBasedOnQuality(rankId, &quality, &clusters, allPoints, &success);
		if (MASTER_ID == rankId)
		{
			fflush(stdout);
			printf("RankId: %d, Time: %d, kMeansIteration: %d\n", rankId, timeIndex, kMeansIteration);
			fflush(stdout);
			printf("quality: %lf\n", quality);
			fflush(stdout);
		}
		timeIndex++;
	}

	if (MASTER_ID == rankId)
	{
		WriteToFile("D:\\Kmeans_Maayan - EOD 20-11-18\\Kmeans_Maayan\\output.txt", clusters, (timeIndex-1)*params.timeInterval, quality);
		timeFinish = MPI_Wtime();
		double totalTime = timeFinish - timeStart;
		fflush(stdout);
		printf("total time: %lf\n", totalTime);
		fflush(stdout);
	}
	MPI_Finalize();
	printf("Finished\n");
	return 0;
}

//Randomly set the first K points to be the centers of K clusters
void initClusters(Cluster ** clusters, Point * points)
{
	int clusterIndex;

	*clusters = (Cluster *)malloc(params.numOfClusters * sizeof(Cluster));

#pragma omp parallel for private(clusterIndex)
	for (clusterIndex = 0; clusterIndex < params.numOfClusters; clusterIndex++)
	{
		Cluster * currentCluster = &((*clusters)[clusterIndex]);
		currentCluster->id = clusterIndex;
		currentCluster->diameter = 0;
		memcpy(currentCluster->center, &points[clusterIndex].location, sizeof(currentCluster->center));
		currentCluster->numOfPoints = 0;
	}
}

double calcDistance(double location1[DIMENSIONS], double location2[DIMENSIONS])
{
	double powDistanceSum = 0;
	int dimensionIndex;
	for (dimensionIndex = 0; dimensionIndex < DIMENSIONS; dimensionIndex++)
	{
		double sub = (location1[dimensionIndex] - location2[dimensionIndex]);
		powDistanceSum += (sub*sub);
//		powDistanceSum += pow((location1[dimensionIndex] - location2[dimensionIndex]), 2);
	}
	double sqrtD = sqrt(powDistanceSum);
	return sqrtD;
}

bool groupPointsHendler(Cluster * clusters, Point ** myPointsOnCpu, Point ** pointsOnGpu, int numOfPoints)
{
	bool ret = groupPointsCuda(clusters, pointsOnGpu, numOfPoints);
	memcpy(*myPointsOnCpu, *pointsOnGpu, numOfPoints * sizeof(Point));
	return ret;

}

void calcClustersCenter(Cluster ** clusters, Point * points)
{
	int clusterIndex;

#pragma omp parallel for private(clusterIndex)
	//for each cluster: calculate the new center
	for (clusterIndex = 0; clusterIndex < params.numOfClusters; clusterIndex++)
	{
		Cluster * currentCluster = &((*clusters)[clusterIndex]);

		int pointIndex;
		int pointsInCluster = 0; // number of points in the current cluster
		double axisSum[DIMENSIONS] = { 0 };
		int axisIndex;
		for (pointIndex = 0; pointIndex < params.numOfPoints; pointIndex++)
		{
			if (isPartOfCluster(*currentCluster, points[pointIndex]))
			{
				for (axisIndex = 0; axisIndex < DIMENSIONS; axisIndex++)
				{
					axisSum[axisIndex] += points[pointIndex].location[axisIndex];
				}
				pointsInCluster++;
			}
		}
		//calc the new cluster center - the avarage of the points location for each axis
		if (pointsInCluster > 0)
		{
			for (axisIndex = 0; axisIndex < DIMENSIONS; axisIndex++)
			{
				currentCluster->center[axisIndex] =
					(axisSum[axisIndex]) / pointsInCluster;
			}
		}
	}
}

void calcSingleDiameter(Cluster * clusterPtr, Point * points)
{
	int p1Index;
	double maxDistance = 0;
#pragma omp parallel for private(p1Index)
	for (p1Index = 0; p1Index < params.numOfPoints; p1Index++)
	{
		if (isPartOfCluster(*clusterPtr, points[p1Index]))
		{
			int p2Index;
#pragma omp parallel for private(p2Index)
			for (p2Index = 0; p2Index < params.numOfPoints; p2Index++)
			{
				if (isPartOfCluster(*clusterPtr, points[p2Index]))
				{
					maxDistance = MAX(maxDistance, calcDistance(points[p1Index].location, points[p2Index].location));
				}
			}
		}
	}
	clusterPtr->diameter = maxDistance;
}

void calcClustersDiameter(Cluster ** clusters, Point * points)
{
	int clusterIndex;
#pragma omp parallel for
	for (clusterIndex = 0; clusterIndex < params.numOfClusters; clusterIndex++)
	{
		Cluster * currentCluster = &((*clusters)[clusterIndex]);
		calcSingleDiameter(currentCluster, points);
	}
}


bool isSameCluster(Point * p1, Point * p2)
{
	return p1->currentCluster == p2->currentCluster;
}

void resetDiameters(Cluster ** clusters)
{
	int clusterIndex;
#pragma omp parallel for private(clusterIndex)
	for (clusterIndex = 0; clusterIndex < params.numOfClusters; clusterIndex++)
	{
		Cluster * currentCluster = &((*clusters)[clusterIndex]);
		currentCluster->diameter = 0;
	}
}

//if the point is a part of the cluster
bool isPartOfCluster(Cluster cluster, Point point)
{
	return point.currentCluster == cluster.id;
}



double calcQuality(Cluster ** clusters, Point * points)
{
	double quality = 0;
	int c1Index;
	int c2Index;
	calcClustersDiameter(clusters, points);

	double factor = (params.numOfClusters) * (params.numOfClusters - 1);


	for (c1Index = 0; c1Index < params.numOfClusters; c1Index++)
	{
		Cluster * firstCluster = &((*clusters)[c1Index]);
		//TODO: PRAGMA?
		for (c2Index = c1Index+1; c2Index < params.numOfClusters; c2Index++)
		{
			if (c1Index != c2Index)
			{
				Cluster * secondCluster = &((*clusters)[c2Index]);
				double distance = calcDistance(firstCluster->center, secondCluster->center);
				quality += ((firstCluster->diameter + secondCluster->diameter) / distance);
			}
		}
	}
	quality /= factor;
	return quality;
}

int getNumOfPointsPerProc(int rankId, Point * points, int numOfProcs)
{
	int numOfPointsPerProc = params.numOfPoints / numOfProcs;
	if (MASTER_ID == rankId)
	{
		numOfPointsPerProc += params.numOfPoints % numOfProcs;
	}

	return numOfPointsPerProc;
}

void allocatePointsArr(Point ** points, int numOfPointsPerProc)
{
	*points = (Point *)malloc(numOfPointsPerProc * sizeof(Point));
}

void determineAlgoTerminate(int rankId, int numOfProcs, int * algoTermination)
{
	MPI_Status status;
	if (MASTER_ID == rankId)
	{
		int procIndex;
		//for each process exclude master
		for (procIndex = 0; procIndex < numOfProcs; procIndex++)
		{
			if (MASTER_ID != procIndex)
			{
				int singleProcAlgoTermination = TRUE;
				MPI_Recv(&singleProcAlgoTermination, 1, MPI_INT, procIndex, 0, MPI_COMM_WORLD, &status);

				if (FALSE == singleProcAlgoTermination)
				{
					*algoTermination = FALSE;
				}
			}
		}
	}
	else
	{
		MPI_Send(algoTermination, 1, MPI_INT, MASTER_ID, 0, MPI_COMM_WORLD);
	}
	MPI_Bcast(algoTermination, 1, MPI_INT, MASTER_ID, MPI_COMM_WORLD);
}

void determineSuccessBasedOnQuality(int rankId, double * quality, Cluster ** clusters, Point * points, int * success)
{
	if (MASTER_ID == rankId)
	{
		*quality = calcQuality(clusters, points);
		if (*quality < params.QM)
		{
			*success = TRUE;
		}
	}
	MPI_Bcast(success, 1, MPI_INT, MASTER_ID, MPI_COMM_WORLD);
}

bool groupPointsSerial(Cluster * clusters, Point ** points, int numOfPoints)
{
	bool ret = true;
	int pointIndex;
	//for each point: find the min distance between the point's location and the clusters centers
#pragma omp parallel for private(pointIndex)
	for (pointIndex = 0; pointIndex < numOfPoints; pointIndex++)
	{
		Point * currentPoint = &((*points)[pointIndex]);
		int clusterIndex = 0;
		double minDistance = calcDistance(currentPoint->location, clusters[clusterIndex].center);
		int closestClusterIndex = clusterIndex;
#pragma omp parallel for private(clusterIndex)
		for (clusterIndex = 1; clusterIndex < params.numOfClusters; clusterIndex++)
		{
			double distance = calcDistance(currentPoint->location, clusters[clusterIndex].center);
			if (distance < minDistance)
			{
				minDistance = distance;
				closestClusterIndex = clusterIndex;
			}
		}
		//update the current cluster
		if (currentPoint->currentCluster != closestClusterIndex)
		{
//			printf("before rankId: %d, pointIndex: %d, currentCluster = %d, closestClusterIndex = %d\n", rankId, pointIndex, currentPoint->currentCluster, closestClusterIndex);
			currentPoint->currentCluster = closestClusterIndex;
			ret = false;
		}
	}

	return ret;
}



