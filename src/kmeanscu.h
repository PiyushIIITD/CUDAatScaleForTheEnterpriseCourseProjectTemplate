#ifndef __KMEANSCU_H__
#define __KMEANSCU_H__
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <stdio.h>
#include <cuda_runtime.h>
using namespace std;

struct Point {
    float x, y;
    int cluster;
};

class Centroid {
public:
    float x, y;
    Centroid() {
        x = 0;
        y = 0;
    }
};
void kmeans(vector<Point>& h_points, vector<Centroid>&h_centroids, int iterations);
#endif