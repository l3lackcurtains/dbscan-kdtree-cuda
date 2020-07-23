/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% by: Alireza Ahmadi                                     %
% University of Bonn- MSc Robotics & Geodetic Engineering%
% Alireza.Ahmadi@uni-bonn.de                             %
% AlirezaAhmadi.xyz                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <iostream> 
#include <vector> 

// KDtree dimension
#define KNN 		4
#define MAX_DIM 	3
#define expWeight 	false
#define NODE_NUM    300
#define dgw 		1
#define EPSILLON  	1e-4

// test 300, 16, 0.5, LAMBDA 0.2 
namespace DynaMap{

	struct kdNode{
		int id;
		float x[MAX_DIM];
		struct kdNode *left, *right;
		float distance;
	};

	class kdTree{
	  public:
		__device__ __host__
		kdTree();
		__device__ __host__
		virtual ~kdTree();
		void init(int querySize);
		void Free(void);
		__device__ __host__
		inline float dist(struct kdNode *a, struct kdNode *b, int dim);
		inline void swap(struct kdNode *x, struct kdNode *y);
		struct kdNode* findMedian(struct kdNode *start, struct kdNode *end, int idx);
		struct kdNode* buildTree(struct kdNode *t, int len, int i, int dim);
		__device__ __host__
		void findNearest(struct kdNode *root, 
						 struct kdNode *nd, 
						 int i, 
						 int dim,
						 struct kdNode **best, 
						 float *best_dist);
		
		__device__ __host__
		std::vector<int> rangeSearch(struct kdNode *root, struct kdNode *rangeLower,
                 struct kdNode *rangeUpper, int dim);
		int visited;
		float *kdDistnaces;
		struct kdNode *kdRoot, *kdQuery, *kdFound, *VisitedNodes;
	};		
}   // namespace DynaMap















