#include "kdtree.h"

namespace DynaMap {
__device__ __host__ kdTree::kdTree() {}
__device__ __host__ kdTree::~kdTree() {}
void kdTree::init(int querySize) {
  visited = 0;
  cudaMallocManaged(&kdDistnaces, sizeof(float));
  cudaMallocManaged(&kdFound, sizeof(struct kdNode));
  cudaMallocManaged(&kdRoot, sizeof(struct kdNode) * NODE_NUM);
  cudaMallocManaged(&kdQuery, sizeof(struct kdNode) * NODE_NUM);
  cudaMallocManaged(&VisitedNodes, sizeof(struct kdNode) * NODE_NUM);
  cudaDeviceSynchronize();
}

void kdTree::Free(void) {
  cudaDeviceSynchronize();
  cudaFree(kdDistnaces);
  cudaFree(kdFound);
  cudaFree(kdRoot);
  cudaFree(kdQuery);
  cudaFree(VisitedNodes);
}
__device__ __host__ inline float kdTree::dist(struct kdNode *a,
                                              struct kdNode *b, int dim) {
  float t, d = 0;
  while (dim--) {
    t = a->x[dim] - b->x[dim];
    d += t * t;
  }
  return d;
}
inline void kdTree::swap(struct kdNode *x, struct kdNode *y) {
#if defined(__CUDA_ARCH__)
  struct kdNode *tmp;
  cudaMallocManaged(&tmp, sizeof(struct kdNode));

  cudaMemcpy(tmp, x, sizeof(struct kdNode), cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();

  cudaMemcpy(x, y, sizeof(struct kdNode), cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();

  cudaMemcpy(y, tmp, sizeof(struct kdNode), cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
#else
  float tmp[MAX_DIM];
  int id;
  memcpy(tmp, x->x, sizeof(tmp));
  id = x->id;

  memcpy(x->x, y->x, sizeof(tmp));
  x->id = y->id;

  memcpy(y->x, tmp, sizeof(tmp));
  y->id = id;
#endif
}
struct kdNode *kdTree::findMedian(struct kdNode *start, struct kdNode *end,
                                  int idx) {
  if (end <= start) return NULL;
  if (end == start + 1) return start;

  struct kdNode *p, *store, *md = start + (end - start) / 2;
  float pivot;
  while (1) {
    pivot = md->x[idx];

    swap(md, end - 1);
    for (store = p = start; p < end; p++) {
      if (p->x[idx] < pivot) {
        if (p != store) swap(p, store);
        store++;
      }
    }
    swap(store, end - 1);

    /* median has duplicate values */
    if (store->x[idx] == md->x[idx]) return md;

    if (store > md)
      end = store;
    else
      start = store;
  }
}
struct kdNode *kdTree::buildTree(struct kdNode *t, int len, int i, int dim) {
  struct kdNode *n;

  if (!len) return 0;

  if ((n = findMedian(t, t + len, i))) {
    i = (i + 1) % dim;
    n->left = buildTree(t, n - t, i, dim);
    n->right = buildTree(n + 1, t + len - (n + 1), i, dim);
  }
  return n;
}

std::vector<int> kdTree::rangeSearch(struct kdNode *root, struct kdNode *rangeLower,
                         struct kdNode *rangeUpper, int dim) {
  std::vector<kdNode *> s;
  kdNode *curr = root;

  std::vector<int> points;

  while (curr != NULL || s.empty() == false) {
    while (curr != NULL) {
      s.push_back(curr);
      curr = curr->left;
    }

    curr = s.back(); 
    s.pop_back();

     if (curr->x[0] >= rangeLower->x[0] && curr->x[0] <= rangeUpper->x[0] &&
        curr->x[1] >= rangeLower->x[1] && curr->x[1] <= rangeUpper->x[1]) {
      points.push_back(curr->id);
    }


    curr = curr->right;
  }

  return points;
}

}  // namespace DynaMap

using namespace DynaMap;
int main() {
  
  int data_size = 8;

  struct kdNode *wp = (struct kdNode*)malloc(sizeof(struct kdNode) * data_size);

  wp[0] = {1, {2, 3}};
  wp[1] = {2, {9, 6}};
  wp[2] = {3, {4, 7}};
  wp[3] = {4, {8, 1}};
  wp[4] = {5, {7, 2}};
  wp[5] = {6, {5, 6}};
  wp[6] = {7, {5, 5}};
  wp[7] = {8, {1, 5}};

  struct kdNode *root;

  struct kdNode rangeLower = {0, {7, 5}};
  struct kdNode rangeUpper = {0, {10, 8}};

  kdTree kd = kdTree();

  root = kd.buildTree(wp, data_size, 0, 2);

  std::vector<int> points = kd.rangeSearch(root, &rangeLower, &rangeUpper, 2);

  for(int i = 0; i < points.size(); i++) {
    printf("%d\n", points[i]);
  }

  return 0;
}
