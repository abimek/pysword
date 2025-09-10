#include "gputensor.h"
#include "ops.h"
#include "cuda_runtime.h"

struct CallData {
  dim3 numBlocks;
  dim3 threadsPerBlock;
};

CallData build_call_data(size_t rows, size_t cols) {
  size_t n = rows * cols;
  dim3 threadsPerBlock(32, 32);
  return CallData {
    dim3((n+threadsPerBlock.x-1) / threadsPerBlock.x, (n+threadsPerBlock.y-1) / threadsPerBlock.y),
    threadsPerBlock,
  };
}

__global__ void transpose(float* a, float* b, size_t rows, size_t cols) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  if (row < rows && col < cols) {
    float sum = 0;
    b[col * rows + row] = a[row*cols+col];
  }
}

void call_transpose(float* a, float* b, size_t a_dim, size_t b_dim) {
  CallData callData = build_call_data(a_dim, b_dim);
  transpose<<<callData.numBlocks, callData.threadsPerBlock>>>(a, b, a_dim, b_dim);
}

__global__ void matmult(float* a, float* b, float* c, size_t rows, size_t cols, size_t n_dim) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  if (row < rows && col < cols) {
    float sum = 0;
    for (int m = 0; m < n_dim; ++m) {
      sum += a[row * n_dim + m] * b[col + cols * m];
    }
    c[row * cols + col] = sum;
  }
}

void call_matmult(float* a, float* b, float* c, size_t a_dim, size_t b_dim, size_t n_dim) {
  CallData callData = build_call_data(a_dim, b_dim);
  matmult<<<callData.numBlocks, callData.threadsPerBlock>>>(a, b, c, a_dim, b_dim, n_dim);
}


/**
* peacewise multiplication 
*/
__global__ void p_mult(float* a, float* b, float* c, size_t rows, size_t cols) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int idx = j * cols + i;
  if (i < cols && j < rows) {
    c[idx] = a[idx] * b[idx];
  }
}

void call_p_mult(float* a, float* b, float* c, size_t rows, size_t cols) {
  CallData callData = build_call_data(rows, cols);
  p_mult<<<callData.numBlocks, callData.threadsPerBlock>>>(a, b, c, rows, cols);
}

/**
* peacewise addition
*/
__global__ void p_add(float* a, float* b, float* c, size_t rows, size_t cols) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int idx = j * cols + i;
  if (i < cols && j < rows) {
    c[idx] = a[idx] + b[idx];
  }
}

void call_p_add(float* a, float* b, float* c, size_t rows, size_t cols) {
  CallData callData = build_call_data(rows, cols);
  p_add<<<callData.numBlocks, callData.threadsPerBlock>>>(a, b, c, rows, cols);
}

/**
* runs scalar multiplication on a 2d array
*/
__global__ void k_mult(float* a, float* b, size_t rows, size_t cols, float k) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = cols * j + i;
  if (i < cols && j < rows) {
    b[idx] = k * a[idx];
  }
}

void call_k_mult(float* a, float* b, size_t rows, size_t cols, float k) {
  CallData callData = build_call_data(rows, cols);
  k_mult<<<callData.numBlocks, callData.threadsPerBlock>>>(a, b, rows, cols, k);
}

/*
* Somewhat efficient sum of elements. This works off the assumption that it's a 1d vector.
*/
__global__ void sum(float* a, float* b, size_t N) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int threads = gridDim.x * blockDim.x;
  float temp_store = 0.0;
  for(int i = idx; i < N; i += threads) temp_store += a[i];
  if (temp_store != 0.0) {
    atomicAdd(b, temp_store);
  }
}

void call_sum(float* a, float* b, size_t N) {
  sum<<<256, 1024>>>(a, b, N);
}