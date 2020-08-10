#include "kdtree.h"

int main(int argc, char ** argv) {
    char inputFname[500];
    if (argc != 2) {
        fprintf(stderr, "Please provide the dataset file path in the arguments\n");
        exit(0);
    }

    // Get the dataset file name from argument
    strcpy(inputFname, argv[1]);
    printf("Using dataset file %s\n", inputFname);

    double * importedDataset =
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

    struct kdNode * wp =
        (struct kdNode *)malloc(sizeof(struct kdNode) * DATASET_COUNT);

    for (int i = 0; i < DATASET_COUNT; i++) {
        wp[i].id = i;
        for (int j = 0; j < DIMENSION; j++) {
            wp[i].x[j] = importedDataset[i * DIMENSION + j];
        }
    }

    struct kdNode * root, * root2;

    double searchPoint[DIMENSION] ={
        131.229813,
        43.925762
    };

    kdTree kd = kdTree();

    root2 = kd.makeTree(wp, DATASET_COUNT, 0);

    inOrderNoRecursion(root2);

    printf("=================================\n");
    root = kd.buildTree(wp, DATASET_COUNT, 0);
    inOrderNoRecursion(root);

    std::vector < int > points = kd.rangeSearch(root, searchPoint);

    for (int i = 0; i < points.size(); i++) {
        printf("%d ", points[i]);
    }

    printf("\n");

    return 0;
}

kdTree::kdTree() {}
kdTree::~kdTree() {}

void kdTree::swap(struct kdNode * x, struct kdNode * y) {
    double tmp[DIMENSION];
    int id;

    for (int i = 0; i < DIMENSION; i++) {
        tmp[i] = x->x[i];
    }
    id = x->id;

    for (int i = 0; i < DIMENSION; i++) {
        x->x[i] = y->x[i];
    }
    id = x->id;
    x->id = y->id;

    for (int i = 0; i < DIMENSION; i++) {
        y->x[i] = tmp[i];
    }
    y->id = id;
}
struct kdNode * kdTree::findMedian(struct kdNode * start, struct kdNode * end,
    int idx) {
    if (end <= start) return NULL;
    if (end == start + 1) return start;

    struct kdNode * p, * store, * md = start + (end - start) / 2;
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

        if (store->x[idx] == md->x[idx]) return md;

        if (store > md)
            end = store;
        else
            start = store;
    }
}

struct kdNode * kdTree::makeTree(struct kdNode * t, int len, int i) {
    struct kdNode * n;
    if (!len) return 0;
    if ((n = findMedian(t, t + len, i))) {
        i = (i + 1) % DIMENSION;
        n->left = makeTree(t, n - t, i);
        n->right = makeTree(n + 1, t + len - (n + 1), i);
    }

    return n;
}

struct kdNode * kdTree::buildTree(struct kdNode * t, int len, int i) {
    struct kdNode * x, * root;
    std::stack < kdNode * > s;

    int start = 0;
    int end = len;

    x = findMedian(t + start, t + end, i);
    int last_mid;;
    root = x;

    while (root != NULL) {
        last_mid = x - t;
        end = last_mid - start;
        i = (i + 1) % DIMENSION;
        if (end >= start) {
            x = findMedian(t + start, t + end, i);
            root->left = x;
        }
        else {
            root->left = NULL;
        }
        end = start + last_mid;
        start = last_mid + 1;

        if (start <= end) {
            root->right = x;
        }
        else {
            break;
        }
        if (root->left != NULL) {
            root = root->left;
        }
    }
    return root;
}

std::vector < int > kdTree::rangeSearch(struct kdNode * root,
    double searchPoint[DIMENSION]) {
    double upperPoint[DIMENSION];

    double lowerPoint[DIMENSION];

    for (int i = 0; i < DIMENSION; i++) {
        upperPoint[i] = searchPoint[i] + EPS;
        lowerPoint[i] = searchPoint[i] - EPS;
    }

    std::vector < int > points ={};

    while (root) {
        if (root->left == NULL) {
            if (root->x[0] >= lowerPoint[0] && root->x[0] <= upperPoint[0] &&
                root->x[1] >= lowerPoint[1] && root->x[1] <= upperPoint[1]) {
                points.push_back(root->id);
            }
            root = root->right;
        }
        else {
            struct kdNode * current = root->left;
            while (current->right && current->right != root)
                current = current->right;
            if (current->right == root) {
                current->right = NULL;
                root = root->right;
            }
            else {
                if (root->x[0] >= lowerPoint[0] && root->x[0] <= upperPoint[0] &&
                    root->x[1] >= lowerPoint[1] && root->x[1] <= upperPoint[1]) {
                    points.push_back(root->id);
                }
                current->right = root;
                root = root->left;
            }
        }
    }
    return points;
}

void inOrderNoRecursion(struct kdNode * curr) {
    std::stack < struct kdNode * > s;

    while (true) {
        while (curr != NULL) {
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

int ImportDataset(char
    const * fname, double * dataset) {
    FILE * fp = fopen(fname, "r");
    if (!fp) {
        printf("Unable to open file\n");
        return (1);
    }

    char buf[4096];
    unsigned long int cnt = 0;
    while (fgets(buf, 4096, fp) && cnt < DATASET_COUNT * DIMENSION) {
        char * field = strtok(buf, ",");
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