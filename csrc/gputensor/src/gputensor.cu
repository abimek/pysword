#include <pybind11/pybind11.h>
#include "gputensor.h"

/**
 * Simple Test, adding two methods
 */
int add(int i, int j) {
	return i + j;

}

__global__ void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

float run() {
  int N = 1;
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  // Perform SAXPY on 1M elements
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

	// Could will throw an error when running on arch without gpu -> lack of gpu
	// to run on
	if (cudaGetLastError() != cudaSuccess) {
		cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(d_x);
		cudaFree(d_y);
		return -1;
	}


  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_x);
  cudaFree(d_y);
	return x[0] + y[0];
  free(x);
  free(y);
}
