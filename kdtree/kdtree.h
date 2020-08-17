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


#include<bits/stdc++.h>


#define DATASET_COUNT 100
#define DIMENSION 2
#define EPS 1.5

struct __align__(8) kdNode{
  int id;
  double x[DIMENSION];
  kdNode *left, *right;
};

class kdTree {
 public:
 struct kdNode *kdRoot;
  __device__ __host__ kdTree();
  __device__ __host__ virtual ~kdTree();
  __device__ __host__ inline  void swap(kdNode *x, kdNode *y);
  __device__ __host__ kdNode *findMedian(kdNode *start, kdNode *end, int idx);
  kdNode *buildTree(kdNode *t, int len, int i);
  kdNode *makeTree(kdNode *t, int len, int i);
  __device__ __host__ kdNode getKdRoot();
  std::vector<int> rangeSearch(kdNode *root,
                                       double searchPoint[DIMENSION]);
  void insert(struct kdNode * t);
};

void inOrderNoRecursion(struct kdNode *curr);
int ImportDataset(char const *fname, double *dataset);

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}
