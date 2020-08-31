#include <bits/stdc++.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <vector>

#define DATASET_COUNT 100
#define DIMENSION 2
#define PARTITION 5
using namespace std;

struct dataNode {
  int id;
  double x[DIMENSION];
  struct dataNode *child;
};

struct IndexStructure {
  int level;
  double range[2];
  struct IndexStructure * buckets[PARTITION];
  struct dataNode *dataRoot;
};

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

  double maxPoints[DIMENSION];
  double minPoints[DIMENSION];

  for (int j = 0; j < DIMENSION; j++) {
    maxPoints[j] = 0;
    minPoints[j] = 999999999;
  }

  for (int i = 0; i < DATASET_COUNT; i++) {
    for (int j = 0; j < DIMENSION; j++) {
      if (importedDataset[i * DIMENSION + j] > maxPoints[j]) {
        maxPoints[j] = importedDataset[i * DIMENSION + j];
      }
      if (importedDataset[i * DIMENSION + j] < minPoints[j]) {
        minPoints[j] = importedDataset[i * DIMENSION + j];
      }
    }
  }

  double advanceRangeBy[DIMENSION];

  for (int j = 0; j < DIMENSION; j++) {
    advanceRangeBy[j] = (maxPoints[j] - minPoints[j]) / PARTITION;
  }

  for (int j = 0; j < DIMENSION; j++) {
    printf("DIM: %d\n", j);
    printf("Max: %f\n", maxPoints[j]);
    printf("Min: %f\n", minPoints[j]);
    printf("Range Advanced: %f\n", advanceRangeBy[j]);
  }

  // Create a indexing structure
  struct IndexStructure indexRoot;
  stack<struct IndexStructure> indexStacked;
  indexStacked.push(indexRoot);


  for (int j = 0; j < DIMENSION; j++) {
    stack<struct IndexStructure> childStacked;

    while (indexStacked.size() > 0) {
      struct IndexStructure * currentIndex;
      currentIndex = &(indexStacked.top());
      indexStacked.pop();
      
      double rightPoint = maxPoints[j];
      for (int k = PARTITION - 1; k >= 0; k--) {
        struct IndexStructure currentBucket;
        currentBucket.range[1] = rightPoint;
        currentBucket.range[0] = rightPoint - advanceRangeBy[j];
        rightPoint = rightPoint - advanceRangeBy[j];
        currentIndex->buckets[k] = &currentBucket;
        childStacked.push(currentBucket);
      }
    }
    while(childStacked.size() > 0) {
      indexStacked.push(childStacked.top());
      childStacked.pop();
    }
  }

  struct IndexStructure * currentBucket = indexRoot.buckets[0];
  printf("Index: %f %f\n", currentBucket->range[0], currentBucket->range[1]);

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
