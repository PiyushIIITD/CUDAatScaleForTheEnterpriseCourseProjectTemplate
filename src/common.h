#ifndef COMMON_H
#define COMMON_H
#define N 10000
#define K 5
#define MAX_ITERS 100
struct __align__(16) Point {
    float x, y;
    int cluster;
    float intensity;
    __host__ __device__
    Point() : x(0), y(0), cluster(-1) {}
};
#endif
