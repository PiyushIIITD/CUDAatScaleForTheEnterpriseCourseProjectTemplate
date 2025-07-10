#include "kmeanscu.h"

#define THREADS_PER_BLOCK 256
const int screenWidth = 800;
const int screenHeight = 600;
const int Npoints = 1000;

vector< Point>pointVec;
vector< Centroid>centroids;
int kvalue = 3;
__device__ float distance(const Point& p, const Centroid& c)
{
    return sqrtf((p.x - c.x) * (p.x - c.x) + (p.y - c.y) * (p.y - c.y));
}

__global__ void clusters(Point* points, Centroid* centroids, int npoints, int k)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;
    int i = ty * gridDim.x * blockDim.x + tx;
    if (i >= npoints) return;

    float minDist = 999999;
    int bestCluster = 0;

    for (int cl = 0; cl < k; cl++) {
        float dist = distance(points[i], centroids[cl]);
        if (dist < minDist) {
            minDist = dist;
            bestCluster = cl;
        }
    }

    points[i].cluster = bestCluster;
}

void kmeans(vector<Point>& h_points, vector< Centroid>& h_centroids,int iterations){
    int npoints = h_points.size();
    int k = h_centroids.size();

    Point* d_points;
    Centroid* d_centroids;

    cudaMalloc(&d_points, npoints * sizeof(Point));
    cudaMemcpy(d_points, h_points.data(), npoints * sizeof(Point), 
    					cudaMemcpyHostToDevice);
    cudaMalloc(&d_centroids, k * sizeof(Centroid));

    for (int it = 0; it < iterations; it++) {
        cudaMemcpy(d_centroids, h_centroids.data(), k * sizeof(Centroid), 
				      cudaMemcpyHostToDevice);

        int nBlocks = ceil( (float) npoints / THREADS_PER_BLOCK );
        clusters<<< nBlocks, THREADS_PER_BLOCK>>>(d_points, d_centroids,npoints,k);
        cudaDeviceSynchronize();

        cudaMemcpy(h_points.data(), d_points, npoints * sizeof(Point), 
					cudaMemcpyDeviceToHost);

        int counts[k];
        fill(counts, counts+k, 0);      //initialize cluster counts to 0
        Centroid new_centroids[k];      
        for (int j = 0; j < npoints; j++) {
            int cl = h_points[j].cluster;
            new_centroids[cl].x += h_points[j].x;
            new_centroids[cl].y += h_points[j].y;
            ++counts[cl];
        }

        for (int c = 0; c < k; c++) {
            if (counts[c] > 0) {
                h_centroids[c].x = new_centroids[c].x / counts[c];
                h_centroids[c].y = new_centroids[c].y / counts[c];
            }
        }
    }

    cudaFree(d_points);
    cudaFree(d_centroids);
}

int doKmeans(vector < Point> &points, vector< Centroid> & centroids)
{
    srand(time(0));
    int npoints = Npoints;
    int k = kvalue;
    int iterations = 30;

    pointVec.resize(npoints);
    for(int i = 0; i < npoints; i++) {
        pointVec[i].x = rand() % screenWidth;
        pointVec[i].y = rand() % screenHeight ;
        pointVec[i].cluster = rand() % k;
    }

    centroids.resize( k );
    for (int i = 0; i < k; ++i) {
        centroids[i].x = pointVec[i].x;
        centroids[i].y = pointVec[i].y;
    }

    kmeans(pointVec, centroids, iterations);

    return 0;
}

int main()
{
   doKmeans(pointVec, centroids);

   //print out every thirty points       
   int newRow = 0;
   for(int i = 0; i < Npoints; i+=30 ) {
      printf("(%3.0f, %3.0f): %d\t", pointVec[i].x, pointVec[i].y, pointVec[i].cluster);
      newRow++;
      if (newRow % 4 == 0) printf("\n");
   }

   printf("\nCentroids: \n");
   for(int i = 0; i < centroids.size(); i++)
       printf("%5.2f, %5.2f\n", centroids[i].x, centroids[i].y);


   return 0;
}