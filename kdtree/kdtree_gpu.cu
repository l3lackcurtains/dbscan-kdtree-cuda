#include "kdtree.h"


#define THREAD_BLOCKS 512
#define THREAD_COUNT 1024

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


kdTree::kdTree() {}
kdTree::~kdTree() {}

double kdTree::dist(kdNode *a, kdNode *b) {
  double t, d = 0;
  int dim = DIMENSION;
  while (dim--) {
    t = a->x[dim] - b->x[dim];
    d += t * t;
  }
  return d;
}

void kdTree::swap(kdNode *x, kdNode *y) {
  double tmp[DIMENSION];
  int id;
  memcpy(tmp, x->x, sizeof(tmp));
  id = x->id;

  memcpy(x->x, y->x, sizeof(tmp));
  x->id = y->id;

  memcpy(y->x, tmp, sizeof(tmp));
  y->id = id;
}
kdNode *kdTree::findMedian(kdNode *start, kdNode *end,
                                  int idx) {
  if (end <= start) return NULL;
  if (end == start + 1) return start;

  kdNode *p, *store, *md = start + (end - start) / 2;
  double pivot;
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

kdNode *kdTree::buildTree(kdNode *t, int len, int i) {
  kdNode *n;

  if (!len) return 0;

  if ((n = findMedian(t, t + len, i))) {
    i = (i + 1) % DIMENSION;
    n->left = buildTree(t, n - t, i);
    n->right = buildTree(n + 1, t + len - (n + 1), i);
  }
  return n;
}

std::vector<int> kdTree::rangeSearch(kdNode *root,
                                     double searchPoint[DIMENSION]) {
  double upperPoint[DIMENSION] ;

  double lowerPoint[DIMENSION];

  for(int i = 0; i < DIMENSION; i++) {
    upperPoint[i] = searchPoint[i] + EPS;
    lowerPoint[i] = searchPoint[i] - EPS;
  }

  std::vector<kdNode *> s;
  kdNode *curr = root;

  std::vector<int> points = {};

  while (curr != NULL || s.empty() == false) {
    while (curr != NULL) {
      s.push_back(curr);
      curr = curr->left;
    }

    curr = s.back();
    s.pop_back();

    if (curr->x[0] >= lowerPoint[0] && curr->x[0] <= upperPoint[0] &&
        curr->x[1] >= lowerPoint[1] && curr->x[1] <= upperPoint[1]) {
      points.push_back(curr->id);
    }
    curr = curr->right;
  }

  return points;
}



__global__ void DBSCAN();

int ImportDataset(char const *fname, double *dataset);
int main(int argc, char **argv) {
  char inputFname[500];
  if (argc != 2) {
    fprintf(stderr, "Please provide the dataset file path in the arguments\n");
    exit(0);
  }

  // Get the dataset file name from argument
  strcpy(inputFname, argv[1]);
  printf("Using dataset file %s\n", inputFname);

  double *importedDataset =
      (double *)malloc(sizeof(double) * DATASET_COUNT * DIMENSION);

  // Import data from dataset
  int ret = ImportDataset(inputFname, importedDataset);
  if (ret == 1) {
    printf("\nError importing the dataset");
    return 0;
  }

  // Check if the data parsed is correct
  for (int i = 0; i < 4; i++) {
    printf("Sample Data %f\n", importedDataset[i]);
  }

  kdNode *wp =
      (kdNode *)malloc(sizeof(kdNode) * DATASET_COUNT);

  for (int i = 0; i < DATASET_COUNT; i++) {
    wp[i].id = i;
    for (int j = 0; j < DIMENSION; j++) {
      wp[i].x[j] = importedDataset[i * DIMENSION + j];
    }
  }

  kdNode *root;

  double searchPoint[DIMENSION] = { 131.145309,44.29855};

  kdTree kd = kdTree();

  root = kd.buildTree(wp, DATASET_COUNT, 0);

  DBSCAN<<<dim3(THREAD_BLOCKS, 1), dim3(THREAD_COUNT, 1)>>>();


  std::vector<int> points = kd.rangeSearch(root, searchPoint);

  // for (int i = 0; i < points.size(); i++) {
  //   printf("%d\n", points[i]);
  // }


  return 0;
}

int ImportDataset(char const *fname, double *dataset) {
  FILE *fp = fopen(fname, "r");
  if (!fp) {
    printf("Unable to open file\n");
    return (1);
  }

  char buf[4096];
  unsigned long int cnt = 0;
  while (fgets(buf, 4096, fp) && cnt < DATASET_COUNT * DIMENSION) {
    char *field = strtok(buf, ",");
    long double tmp;
    sscanf(field, "%Lf", &tmp);
    dataset[cnt] = tmp;
    cnt++;

    while (field) {
      field = strtok(NULL, ",");

      if (field != NULL) {
        long double tmp;
        sscanf(field, "%Lf", &tmp);
        dataset[cnt] = tmp;
        cnt++;
      }
    }
  }
  fclose(fp);
  return 0;
}

__global__ void DBSCAN() {
  
  printf("Test");

}