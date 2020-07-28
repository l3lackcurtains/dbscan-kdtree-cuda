/*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% by: Alireza Ahmadi                                     %
% University of Bonn- MSc Robotics & Geodetic Engineering%
% Alireza.Ahmadi@uni-bonn.de                             %
% AlirezaAhmadi.xyz                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*/
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <iostream>
#include <vector>


#define DATASET_COUNT 1000000
#define DIMENSION 2
#define EPS 1.5

namespace DynaMap {

struct kdNode {
  int id;
  double x[DIMENSION];
  struct kdNode *left, *right;
};

class kdTree {
 public:
  kdTree();

  virtual ~kdTree();

  double dist(struct kdNode *a, struct kdNode *b);
  void swap(struct kdNode *x, struct kdNode *y);
  struct kdNode *findMedian(struct kdNode *start, struct kdNode *end, int idx);
  struct kdNode *buildTree(struct kdNode *t, int len, int i);

  std::vector<int> kdTree::rangeSearch(struct kdNode *root,
                                       double searchPoint[DIMENSION]);
};
}  // namespace DynaMap
