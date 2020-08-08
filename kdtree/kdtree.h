#include <cuda.h>
#include <cuda_runtime.h>
#include <host_defines.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <iostream>
#include <vector>


#define DATASET_COUNT 10000
#define DIMENSION 2
#define EPS 1.5

struct __align__(8) kdNode{
  int id;
  double x[DIMENSION];
  kdNode *left, *right;
};

class kdTree {
private:
  kdNode *kdRoot;
 public:
  __device__ __host__ kdTree();
  __device__ __host__ virtual ~kdTree();
  __device__ __host__ inline  double dist(kdNode *a, kdNode *b);
  __device__ __host__ inline  void swap(kdNode *x, kdNode *y);
  __device__ __host__ kdNode *findMedian(kdNode *start, kdNode *end, int idx);
  kdNode *buildTree(kdNode *t, int len, int i);
  __device__ __host__ kdNode getKdRoot();
  std::vector<int> rangeSearch(kdNode *root,
                                       double searchPoint[DIMENSION]);
};
