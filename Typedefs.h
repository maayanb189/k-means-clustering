#pragma once

using namespace std;

#define FALSE 0
#define TRUE 1

#define MASTER_ID 0
#define DIMENSIONS 3
#define MAX(X,Y)	(X>Y?X:Y)
#define ABS(X)	(X<0 ? X*(-1) : X)

#define PARALLEL

struct Cluster
{
	int id;
	double diameter;
	double center[DIMENSIONS];
	int numOfPoints;
};

struct Point
{
	double location[DIMENSIONS];
	double velocity[DIMENSIONS];
	int currentCluster;
};

struct AlgoParams
{
	int numOfPoints; //N
	int numOfClusters; //K
	double endOfTime; //T
	double timeInterval; //dT
	double maxIterations; //LIMIT
	double QM;
};

//#ifdef PARALLEL

extern AlgoParams params;

//#endif