#include "MpiUtils.h"
#include <stdlib.h>
#include "Typedefs.h"
#include <stddef.h>
#include <string.h>

#define DEFAULT_TAG	0
#define SCATTER

int * sendRecvCnts;
int * displs;

MPI_Datatype MPI_PointType;
MPI_Datatype  MPI_ClusterType;
MPI_Datatype  MPI_AlgoParamsType;

void MpiUtils_initProgram(int * argc, char *** argv, int * rankId, int * numOfProcs)
{
	MPI_Init(argc, argv);
	MPI_Comm_rank(MPI_COMM_WORLD, rankId);
	MPI_Comm_size(MPI_COMM_WORLD, numOfProcs);
}

void MpiUtils_createPointType()
{
	//point struct
	MPI_Datatype pointTypes[] = { MPI_DOUBLE, MPI_DOUBLE, MPI_INT };
	int pointBlocklen[] = { DIMENSIONS, DIMENSIONS, 1 };
	MPI_Aint pointDisp[3];

	//points offsets
	pointDisp[0] = offsetof(Point, location);
	pointDisp[1] = offsetof(Point, velocity);
	pointDisp[2] = offsetof(Point, currentCluster);

	//create MPI data type
	MPI_Type_create_struct(3, pointBlocklen, pointDisp, pointTypes, &MPI_PointType);
	MPI_Type_commit(&MPI_PointType);

}

void MpiUtils_createClusterType()
{
	//clustr struct
	MPI_Datatype clusterTypes[] = { MPI_INT, MPI_DOUBLE,MPI_DOUBLE, MPI_INT };
	int clusterBlocklen[] = { 1, 1, DIMENSIONS, 1 };
	MPI_Aint clusterDisp[4];

	//cluster offsets
	clusterDisp[0] = offsetof(Cluster, id);
	clusterDisp[1] = offsetof(Cluster, diameter);
	clusterDisp[2] = offsetof(Cluster, center);
	clusterDisp[3] = offsetof(Cluster, numOfPoints);

	//create MPI data types
	MPI_Type_create_struct(4, clusterBlocklen, clusterDisp, clusterTypes, &MPI_ClusterType);
	MPI_Type_commit(&MPI_ClusterType);
}

void MpiUtils_createAlgoParamsType()
{
	//AlgoParams struct
	MPI_Datatype clusterTypes[] = { MPI_INT, MPI_INT, MPI_DOUBLE,MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };
	int clusterBlocklen[] = { 1, 1, 1, 1, 1, 1 };
	MPI_Aint clusterDisp[6];

	//cluster offsets
	clusterDisp[0] = offsetof(AlgoParams, numOfPoints);
	clusterDisp[1] = offsetof(AlgoParams, numOfClusters);
	clusterDisp[2] = offsetof(AlgoParams, endOfTime);
	clusterDisp[3] = offsetof(AlgoParams, timeInterval);
	clusterDisp[4] = offsetof(AlgoParams, maxIterations);
	clusterDisp[5] = offsetof(AlgoParams, QM);

	//create MPI data types
	MPI_Type_create_struct(6, clusterBlocklen, clusterDisp, clusterTypes, &MPI_AlgoParamsType);
	MPI_Type_commit(&MPI_AlgoParamsType);
}

void MpiUtils_initSendResvParams(int numOfProcs)
{
	sendRecvCnts = (int *)malloc(numOfProcs * sizeof(int));
	displs = (int *)malloc(numOfProcs * sizeof(int));
	int ProcIndex;
	int numOfPointsPerProc = params.numOfPoints / numOfProcs;
	int reminder = params.numOfPoints % numOfProcs;

	sendRecvCnts[MASTER_ID] = numOfPointsPerProc + reminder;
	displs[MASTER_ID] = 0;

	for (ProcIndex = 1; ProcIndex < numOfProcs; ProcIndex++)
	{
		sendRecvCnts[ProcIndex] = numOfPointsPerProc;
		displs[ProcIndex] = ProcIndex*numOfPointsPerProc + reminder;
	}
}
void distributePoints(int rankId, Point * allPoints, Point * myPoints, int numOfProcs)
{

	MPI_Scatterv(allPoints, sendRecvCnts, displs, MPI_PointType,
		myPoints, sendRecvCnts[rankId], MPI_PointType,
		MASTER_ID, MPI_COMM_WORLD);
}

void collectPoints(int rankId, Point * allPoints, Point * myPoints, int numOfProcs)
{

	MPI_Gatherv(myPoints, sendRecvCnts[rankId], MPI_PointType,
		allPoints, sendRecvCnts, displs, MPI_PointType, MASTER_ID, MPI_COMM_WORLD);
}

void MpiUtils_BcastAlgoParams()
{
	MPI_Bcast(&params, 1, MPI_AlgoParamsType, MASTER_ID, MPI_COMM_WORLD);
}

void MpiUtils_BcastClusters(Cluster * clusters)
{
	MPI_Bcast(clusters, params.numOfClusters, MPI_ClusterType, MASTER_ID, MPI_COMM_WORLD);
}

