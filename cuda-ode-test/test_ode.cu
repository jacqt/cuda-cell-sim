#define NUM_DIFF_EQUATIONS 2  // number of differential equations
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

// Differential equations for a circle
__device__ float dy_dt_1(float y1, float y2, float t) {
  return y2;
}

__device__ float dy_dt_2(float y1, float y2, float t) {
  return -y1;
}

// Forwarde ODE solver that moves forward by onestep
__global__ void solve(float* ys, float* next_ys, float time, float timestep, int num_variables) {
  next_ys[0] = ys[0] + timestep * dy_dt_1(ys[0], ys[1], time);
  next_ys[1] = ys[1] + timestep * dy_dt_2(ys[0], ys[1], time);
}

int main(int argc, char* argv[]) {

  float *dev_ys, *dev_next_ys, *temp;
  int array_size = NUM_DIFF_EQUATIONS*sizeof(float);
  cudaMalloc( (void**)&dev_ys, array_size);
  cudaMalloc( (void**)&dev_next_ys, array_size);

  // Create the two input vectors
  float ys[NUM_DIFF_EQUATIONS], next_ys[NUM_DIFF_EQUATIONS];
  ys[0] = 1;
  ys[1] = 0;
  next_ys[0] = 0;
  next_ys[1] = 0;

  cudaMemcpy(dev_ys, ys, array_size, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_next_ys, next_ys, array_size, cudaMemcpyHostToDevice);

  for (int i = 0; i < 20000; i += 10) {
    solve<<<1,1>>>(dev_ys, dev_next_ys, 0, 0.001, NUM_DIFF_EQUATIONS);

    cudaMemcpy(next_ys, dev_next_ys, array_size, cudaMemcpyDeviceToHost);
    cout << "next_ys = [";
    for (int i = 0; i < NUM_DIFF_EQUATIONS; i++) {
      cout << next_ys[i] << ", ";
    }
    cout << "]\n";
    temp = dev_ys;
    dev_ys = dev_next_ys;
    dev_next_ys = temp;

  }

  cudaFree(dev_ys);
  cudaFree(dev_next_ys);
  return 0;
}
