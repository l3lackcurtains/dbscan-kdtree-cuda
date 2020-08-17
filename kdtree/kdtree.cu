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

    // Check if the data parsed is correct
    for (int i = 0; i < 4; i++)
    {
        printf("Sample Data %f\n", importedDataset[i]);
    }

    struct kdNode *wp =
        (struct kdNode *)malloc(sizeof(struct kdNode) * DATASET_COUNT);

    for (int i = 0; i < DATASET_COUNT; i++)
    {
        wp[i].id = i;
        for (int j = 0; j < DIMENSION; j++)
        {
            wp[i].x[j] = importedDataset[i * DIMENSION + j];
        }
    }

    struct kdNode *root, *root2;

    double searchPoint[DIMENSION] = {
        131.229813,
        43.925762};

    kdTree kd = kdTree();

    root = kd.buildTree(wp, DATASET_COUNT, 0);
    inOrderNoRecursion(root);
    
    root2 = kd.makeTree(wp, DATASET_COUNT, 0);
    // inOrderNoRecursion(root2);

    std::vector<int> points = kd.rangeSearch(root, searchPoint);

    for (int i = 0; i < points.size(); i++)
    {
        printf("%d ", points[i]);
    }

    printf("\n");

    return 0;
}

kdTree::kdTree() {}
kdTree::~kdTree() {}


void kdTree::insert(struct kdNode * t) {
    if (kdRoot == NULL) {
        kdRoot = t;
        return;
    }
    int currDim = 0;
    struct kdNode * currRoot = kdRoot;
    while (true) {
        if (t->x[currDim] < currRoot->x[currDim]) {
            if (currRoot->id == t->id) {
                return;
            }
            if (currRoot->left == NULL) {
                currRoot->left = t;
                break;
            } else
                currRoot = currRoot->left;
        } else {
            if (currRoot->right == NULL) {
                currRoot->right = t;
                break;
            } else
                currRoot = currRoot->right;
        }
        currDim = (currDim + 1) % DIMENSION;
    }
    return;
}


struct kdNode * kdTree::makeTree(struct kdNode * t, int len, int i) {
    struct kdNode * n;
    struct kdNode * start;
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