#pragma once
#include "Typedefs.h"

void ReadFromFile(char * fileName, Point ** points);

void WriteToFile(char * fileName, Cluster * clusters, double time, double quality);

void ClustersToFile(char * fileName, Cluster * clusters, int timeIndex);
