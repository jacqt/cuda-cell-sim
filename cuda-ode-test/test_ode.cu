#define NUM_DIFF_EQUATIONS 2  // number of differential equations
#define NUM_ITERATIONS 10000
#define TIME_STEP 0.001
#define cuda_get(matrix, row, column, width) (matrix[(row)*(width) + (column)])

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

// Forward ODE solver that moves forward by onestep
__global__ void solve_one(float* ys, float* next_ys, float time, float timestep) {
  next_ys[0] = ys[0] + timestep * dy_dt_1(ys[0], ys[1], time);
  next_ys[1] = ys[1] + timestep * dy_dt_2(ys[0], ys[1], time);
}

__global__ void solve(float* ys, float time) {
  for (int i = 1; i < NUM_ITERATIONS; ++i) {
    cuda_get(ys, i, 0, 2) = cuda_get(ys, i-1, 0, 2) + TIME_STEP * dy_dt_1(cuda_get(ys, i-1, 0, 2), cuda_get(ys, i-1, 1, 2), time);
    cuda_get(ys, i, 1, 2) = cuda_get(ys, i-1, 1, 2) + TIME_STEP * dy_dt_2(cuda_get(ys, i-1, 0, 2), cuda_get(ys, i-1, 1, 2), time);
    time += TIME_STEP;
  }
}

int main(int argc, char* argv[]) {

  float *dev_ys;
  int dev_ys_size = NUM_ITERATIONS * NUM_DIFF_EQUATIONS * sizeof(float);

  cudaMalloc( (void**)&dev_ys, dev_ys_size);

  // Create the two input vectors
  float ys[NUM_ITERATIONS * NUM_DIFF_EQUATIONS];
  cuda_get(ys, 0, 0, NUM_DIFF_EQUATIONS) = 1;
  cuda_get(ys, 0, 1, NUM_DIFF_EQUATIONS) = 0;

  cudaMemcpy(dev_ys, ys, dev_ys_size, cudaMemcpyHostToDevice);

  solve<<<1,1>>>(dev_ys, 0);

  cudaMemcpy(ys, dev_ys, dev_ys_size, cudaMemcpyDeviceToHost);
  for (int row = 0; row < NUM_ITERATIONS; row += 100) {
    cout << "next_ys[" << row << "] = [";
    for (int i = 0; i < NUM_DIFF_EQUATIONS; i++) {
      cout << cuda_get(ys, row, i, NUM_DIFF_EQUATIONS)  << ", ";
    }
    cout << "]\n";
  }

  cudaFree(dev_ys);
  return 0;
}
