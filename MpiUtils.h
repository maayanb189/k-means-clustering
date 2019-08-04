#pragma once

#include "mpi.h"
#include "Typedefs.h"

void MpiUtils_initProgram(int * args, char *** argv, int * rankId, int * numOfProcs);
void MpiUtils_createPointType();
void MpiUtils_createClusterType();
void MpiUtils_createAlgoParamsType();
void MpiUtils_initSendResvParams(int numOfProcs);
void distributePoints(int rankId, Point * allPoints, Point * points, int numOfProcs);
void collectPoints(int rankId, Point * allPoints, Point * points, int numOfProcs);
void MpiUtils_BcastAlgoParams();
void MpiUtils_BcastClusters(Cluster * clusters);