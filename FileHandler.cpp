#include "FileHandler.h"

#include <stdio.h>
#include <stdlib.h>



void ReadParams(FILE * file);
void ReadPoints(FILE * file, Point ** points);

void ReadFromFile(char * fileName, Point ** points)
{
	FILE * f = fopen(fileName, "r");
	if (f == NULL)
	{
		printf("File Not Found!!\n");
		return;
	}
	ReadParams(f); //Read the first line
	ReadPoints(f, points);
	fclose(f);
}

void ReadParams(FILE * file)
{
	fscanf(file, "%d %d %lf %lf %lf %lf",
		&params.numOfPoints, &params.numOfClusters, &params.endOfTime,
		&params.timeInterval, &params.maxIterations, &params.QM);
}
void ReadPoints(FILE * file, Point ** points)
{
	int pointIndex = 0;

	*points = (Point *)malloc(params.numOfPoints * sizeof(Point));
	if (points == NULL)
	{
		printf("Memory Allocation Error!!\n");
	}

	for (pointIndex = 0; pointIndex < params.numOfPoints; pointIndex++)
	{
		Point * currentPoint = &((*points)[pointIndex]);

		double * location = currentPoint->location;
		double * velocity = currentPoint->velocity;

		fscanf(file, "%lf %lf %lf %lf %lf %lf\n", &location[0], &location[1], &location[2],
			&velocity[0], &velocity[1], &velocity[2]);

		currentPoint->currentCluster = -1;

		fflush(stdout);
	}
}

void WriteToFile(char * fileName, Cluster * clusters, double time, double quality)
{
	int clusterIndex;
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("File Not Found!!\n");
		return;
	}
	fflush(stdout);
	fprintf(f, "First occurrence t = %lf with q = %lf\n", time, quality);
	fprintf(f, "Centers of the clusters:\n");

	for (clusterIndex = 0; clusterIndex < params.numOfClusters; clusterIndex++)
	{
		Cluster * currentCluster = &(clusters[clusterIndex]);
		fprintf(f, "%lf, %lf, %lf\n", currentCluster->center[0], currentCluster->center[1], currentCluster->center[2]);
		fflush(stdout);
	}
	fclose(f);
}

void ClustersToFile(char * fileName, Cluster * clusters, int timeIndex)
{
	static int writeIndex = 0;
	int clusterIndex;
	FILE * f;
	if (0 == writeIndex)
	{
		f = fopen(fileName, "w");
	}
	else
	{
		f = fopen(fileName, "a");
	}

	if (f == NULL)
	{
		printf("File Not Found!!\n");
		return;
	}
	fflush(stdout);
	fprintf(f, "timeIndex = %d \n", timeIndex);
	fprintf(f, "writeIndex = %d \n", writeIndex++);
	fprintf(f, "Centers of the clusters:\n");

	for (clusterIndex = 0; clusterIndex < params.numOfClusters; clusterIndex++)
	{
		Cluster * currentCluster = &(clusters[clusterIndex]);
		fprintf(f, "%lf, %lf, %lf\n", currentCluster->center[0], currentCluster->center[1], currentCluster->center[2]);
		fflush(stdout);
	}
	fclose(f);
}