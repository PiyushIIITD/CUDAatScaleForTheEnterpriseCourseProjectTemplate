#ifndef TIMER_H
#define TIMER_H
#include <cuda_runtime.h>
struct GpuTimer {
  cudaEvent_t start, stop;
  GpuTimer() { cudaEventCreate(&start); cudaEventCreate(&stop); }
  void Start(){ cudaEventRecord(start); }
  void Stop(){ cudaEventRecord(stop); cudaEventSynchronize(stop); }
  float Elapsed(){
    float ms; cudaEventElapsedTime(&ms, start, stop); return ms;
  }
};
#endif
