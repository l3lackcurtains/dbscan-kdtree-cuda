#include "kdtree.h"

int main(int argc, char **argv)
{
    char inputFname[500];
    if (argc != 2)
    {
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
    if (ret == 1)
    {
        printf("\nError importing the dataset");
        return 0;
    }

    datasetPreprocessing(importedDataset);

    // Check if the data parsed is correct
    for (int i = 0; i < 4; i++)
    {
        printf("Sample Data %f\n", importedDataset[i]);
    }

    struct kdNode *wp =
        (struct kdNode *)malloc(sizeof(struct kdNode) * DATASET_COUNT);

    int blocks = 5;
    int divided = DATASET_COUNT / blocks + 1;

    struct kdNode **wp2 =
        (struct kdNode **)malloc(sizeof(struct kdNode*) * blocks);
    for(int j = 0; j < blocks; j++) {
        wp2[j] =
        (struct kdNode *)malloc(sizeof(struct kdNode) * divided);
    }
    

    for (int i = 0; i < DATASET_COUNT; i++)
    {
        wp[i].id = i;
        for (int j = 0; j < DIMENSION; j++)
        {
            wp[i].x[j] = importedDataset[i * DIMENSION + j];
        }
    }
    
    printf("SORTED DATASET\n");
    for (int i = 0; i < DATASET_COUNT; i++)
    {
        printf("%d ",wp[i].id);
        
    }
    printf("\n");

    for(int i = 0; i < blocks; i++) {
        for(int j = 0; j < divided; j++) {
            if(i*divided + j < DATASET_COUNT) {
                wp2[i][j].id = wp[i*divided + j].id;
                for (int k = 0; k < DIMENSION; k++)
                {
                    wp2[i][j].x[k] = wp[i*divided + j].x[k];
                }
            } else {
                wp2[i][j].id = -1;
            }
        }
    }

    printf("DIVIDED DATASET\n");
    for (int i = 0; i < blocks; i++)
    {   
        for(int j = 0; j < divided; j++) {
            printf("%d ",wp2[i][j].id);
        }
        printf("\n");
        
    }
    printf("\n");

    struct kdNode *root =
        (struct kdNode *)malloc(sizeof(struct kdNode) * blocks);
    
    struct kdNode *root2 =
        (struct kdNode *)malloc(sizeof(struct kdNode) * blocks);

    double searchPoint[DIMENSION] = {
        131.229813,
        43.925762
    };

    kdTree kd = kdTree();

    for(int i = 0; i < blocks; i++ ) {
        int len = 0;
        for(int j = 0; j < divided; j++) {
            if(wp2[i][j].id != -1) {
                len++;
            }
        }
        if(len > 0) {
            root[i] = *kd.buildTree(wp2[i], len, 0);
        } else {
            root[i].id = -1;
        }
    }

    int len = 0;
    for(int i = 0; i < blocks; i++ ) {
        if(root[i].id != -1) len++;
    }

    for(int i = 0; i < len; i++ ) {
        root2[i] = root[i];
    }

    printf("TREE ROOTS\n");
    for (int i = 0; i < len; i++)
    {
        printf("%d ",root[i].id);
        
    }
    printf("\n");

    std::vector<int> sec;
    printf("Len: %d\n", len);
    int mid = len/2;
    if(len %2 == 0) mid = mid - 1;
    sec.push_back(root2[mid].id);

    for(int i = mid - 1; i >= 0; i--) {
        sec.push_back(root2[i].id);
    }

    for(int i = mid + 1; i < len; i++) {
        sec.push_back(root2[i].id);
    }

    for(int i = 0; i < sec.size(); i++) {
        printf("sec: %d\n", sec[i]);
    }
    

    struct kdNode *mainRoot = NULL;
    
    for(int k = 0; k < sec.size(); k++) {
        for(int i = 0; i < len; i++) {
            if(root[i].id == sec[k]) {
                struct kdNode *item = &root[i];
                mainRoot = kd.insertRec(mainRoot, item, 0);
                break;
            }
        }
    }
    
    printf("\n========================\n");
    inOrderNoRecursion(mainRoot);
    preOrderNoRecursion(mainRoot);
    printf("========================\n");
    std::vector<int> points = kd.rangeSearch(mainRoot, searchPoint);
    for (int i = 0; i < points.size(); i++)
    {
        printf("%d ", points[i]);
    }
    printf("\n");

    printf("MAIN::\n");
    mainRoot = kd.makeTree(wp, DATASET_COUNT, 0);
    printf("\n========================\n");
    inOrderNoRecursion(mainRoot);
    preOrderNoRecursion(mainRoot);
    printf("========================\n");
    points = kd.rangeSearch(mainRoot, searchPoint);
    printf("\n");
    

    for (int i = 0; i < points.size(); i++)
    {
        printf("%d ", points[i]);
    }
    printf("\n");
    return 0;
}

kdTree::kdTree() {}
kdTree::~kdTree() {}

struct kdNode *kdTree::insertRec(struct kdNode *root, struct kdNode *item, unsigned depth) 
{ 
    if (root == NULL) 
       return item;
  
    unsigned cd = depth % DIMENSION;

    if (item->x[cd] < (root->x[cd]))
        root->left  = insertRec(root->left, item, depth + 1); 
    else
        root->right = insertRec(root->right, item, depth + 1); 
  
    return root; 
} 


struct kdNode * kdTree::makeTree(struct kdNode * t, int len, int i) {
    struct kdNode * n;
    n = findMedian(t, t + len, i);
    if (n) {
        i = (i + 1) % DIMENSION;
        n->left = makeTree(t, n - t, i);
        n->right = makeTree(n + 1, t + len - (n + 1), i);
    }

    return n;
}

struct kdNode * kdTree::buildTree(struct kdNode * t, int len, int i) {
   
    struct kdNode * x, * root, *xy, *currentRoot;
    std::stack < kdNode * > s;

    int start = 0;
    int end = len;
    

    x = findMedian(t + start, t + end, i);
    int last_mid = x - t;
    int last_start;
    int last_end;
    root = x;

    last_start = start;
    last_end = end;
    int left_mid;

    currentRoot = root;
    
    while (currentRoot != NULL) {
        end = last_mid;
        start = last_start;

        i = (i + 1) % DIMENSION;
        x = findMedian(t + start, t + end, i);
        

        currentRoot->left = x;
        if (x != NULL) {
            left_mid = x - t;
        }

        start = last_mid + 1;
        end = last_end;
        
        xy = findMedian(t + start, t + end, i);

        if(xy != NULL) {
            currentRoot->right = xy;
            s.push(xy);
            s.push(start + t);
            s.push(end + t);
        } else {
            currentRoot->right = NULL;
        }

        if (currentRoot->left != NULL) {
            currentRoot = currentRoot->left;
            last_end = last_mid;
            last_mid = left_mid;
        } else {
            if(s.empty()) break;
            if(s.top() != NULL) {
                last_end = s.top() - t;
                s.pop();
            }
            if(s.top() != NULL) {
                last_start = s.top() - t;
                s.pop();
            }

            if(s.top() != NULL) {
                last_mid = s.top() - t;
            }
            currentRoot = s.top();
            x = s.top();
            s.pop();
        }
    }
    return root;
}

void kdTree::swap(struct kdNode *x, struct kdNode *y)
{
    double tmp[DIMENSION];
    int id;

    for (int i = 0; i < DIMENSION; i++)
    {
        tmp[i] = x->x[i];
    }
    id = x->id;

    for (int i = 0; i < DIMENSION; i++)
    {
        x->x[i] = y->x[i];
    }
    id = x->id;
    x->id = y->id;

    for (int i = 0; i < DIMENSION; i++)
    {
        y->x[i] = tmp[i];
    }
    y->id = id;
}
struct kdNode *kdTree::findMedian(struct kdNode *start, struct kdNode *end,
                                  int idx)
{
    if (end <= start)
        return NULL;
    if (end == start + 1)
        return start;

    struct kdNode *p, *store, *md = start + (end - start) / 2;
    double pivot;
    while (1)
    {
        pivot = md->x[idx];

        swap(md, end - 1);
        for (store = p = start; p < end; p++)
        {
            if (p->x[idx] < pivot)
            {
                if (p != store)
                    swap(p, store);
                store++;
            }
        }
        swap(store, end - 1);

        if (store->x[idx] == md->x[idx])
            return md;

        if (store > md)
            end = store;
        else
            start = store;
    }
}

std::vector<int> kdTree::rangeSearch(struct kdNode *root,
                                     double searchPoint[DIMENSION])
{
    double upperPoint[DIMENSION];

    double lowerPoint[DIMENSION];

    for (int i = 0; i < DIMENSION; i++)
    {
        upperPoint[i] = searchPoint[i] + EPS;
        lowerPoint[i] = searchPoint[i] - EPS;
    }

    std::vector<int> points = {};

    while (root)
    {
        if (root->left == NULL)
        {
            if (root->x[0] >= lowerPoint[0] && root->x[0] <= upperPoint[0] &&
                root->x[1] >= lowerPoint[1] && root->x[1] <= upperPoint[1])
            {
                points.push_back(root->id);
            }
            root = root->right;
        }
        else
        {
            struct kdNode *current = root->left;
            while (current->right && current->right != root)
                current = current->right;
            if (current->right == root)
            {
                current->right = NULL;
                root = root->right;
            }
            else
            {
                if (root->x[0] >= lowerPoint[0] && root->x[0] <= upperPoint[0] &&
                    root->x[1] >= lowerPoint[1] && root->x[1] <= upperPoint[1])
                {
                    points.push_back(root->id);
                }
                current->right = root;
                root = root->left;
            }
        }
    }
    return points;
}

void inOrderNoRecursion(struct kdNode *curr)
{
    std::stack<struct kdNode *> s;

    while (true)
    {
        while (curr != NULL)
        {
            s.push(curr);
            curr = curr->left;
        }
        if (s.size() == 0)
            break;
        curr = s.top();
        s.pop();

        printf("%d ", curr->id);
        curr = curr->right;
    }
    printf("\n");
}

void preOrderNoRecursion(struct kdNode *curr)
{
    if (curr == NULL) 
       return; 
    std::stack<struct kdNode *> s; 
    s.push(curr); 
  
    while (s.empty() == false) 
    { 
        struct kdNode *node = s.top(); 
        printf("%d ", node->id);
        s.pop(); 
  
        if (node->right) 
            s.push(node->right); 
        if (node->left) 
            s.push(node->left); 
    }

    printf("\n");
}

std::vector<int> inorderToVector(struct kdNode *curr)
{
    
    std::stack<struct kdNode *> s;
    std::vector<int> d = {};

    if (curr == NULL) 
       return d; 
    s.push(curr); 
    while (s.empty() == false) 
    { 
        struct kdNode *node = s.top(); 
        d.push_back(node->id);
        s.pop(); 
        if (node->right) 
            s.push(node->right); 
        if (node->left) 
            s.push(node->left); 
    }
    return d;
}

void datasetPreprocessing(double *h_dataset) {

  double dataset_tuple_x[DATASET_COUNT];
  double dataset_tuple_y[DATASET_COUNT];
  for (int i = 0; i < DATASET_COUNT; i++) {
    dataset_tuple_x[i] = h_dataset[i * DIMENSION];
    dataset_tuple_y[i] = h_dataset[i * DIMENSION + 1];
  }

  double *d_dataset_tuple1;
  gpuErrchk(cudaMalloc(&d_dataset_tuple1, DATASET_COUNT * sizeof(double)));

  double *d_dataset_tuple2;
  gpuErrchk(cudaMalloc(&d_dataset_tuple2, DATASET_COUNT * sizeof(double)));

  gpuErrchk(cudaMemcpy(d_dataset_tuple1, dataset_tuple_x,
                       DATASET_COUNT * sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_dataset_tuple2, dataset_tuple_y,
                       DATASET_COUNT * sizeof(double), cudaMemcpyHostToDevice));

  thrust::device_ptr<double> dev_ptr_vector1 =
      thrust::device_pointer_cast(d_dataset_tuple1);
  thrust::device_ptr<double> dev_ptr_vector2 =
      thrust::device_pointer_cast(d_dataset_tuple2);

  auto begin = thrust::make_zip_iterator(
      thrust::make_tuple(dev_ptr_vector1, dev_ptr_vector2));
  auto end = thrust::make_zip_iterator(thrust::make_tuple(
      dev_ptr_vector1 + DATASET_COUNT, dev_ptr_vector2 + DATASET_COUNT));

  thrust::sort(begin, end, TupleComp());

  double *h_vector1_output = (double *)malloc(DATASET_COUNT * sizeof(double));
  double *h_vector2_output = (double *)malloc(DATASET_COUNT * sizeof(double));

  gpuErrchk(cudaMemcpy(h_vector1_output, d_dataset_tuple1,
                       DATASET_COUNT * sizeof(double), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(h_vector2_output, d_dataset_tuple2,
                       DATASET_COUNT * sizeof(double), cudaMemcpyDeviceToHost));

  for (int i = 0; i < DATASET_COUNT; i++) {
    h_dataset[i * DIMENSION] = h_vector1_output[i];
    h_dataset[i * DIMENSION + 1] = h_vector2_output[i];
  }
}


int ImportDataset(char const *fname, double *dataset)
{
    FILE *fp = fopen(fname, "r");
    if (!fp)
    {
        printf("Unable to open file\n");
        return (1);
    }

    char buf[4096];
    unsigned long int cnt = 0;
    while (fgets(buf, 4096, fp) && cnt < DATASET_COUNT * DIMENSION)
    {
        char *field = strtok(buf, ",");
        long double tmp;
        sscanf(field, "%Lf", &tmp);
        dataset[cnt] = tmp;
        cnt++;

        while (field)
        {
            field = strtok(NULL, ",");

            if (field != NULL)
            {
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