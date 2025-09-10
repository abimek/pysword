#pragma once
#include "gputensor.h"
#include "cuda_runtime.h"
void call_k_mult(float* a, float* b, size_t rows, size_t cols, float k);
void call_p_add(float* a, float* b, float* c, size_t rows, size_t cols);
void call_p_mult(float* a, float* b, float* c, size_t rows, size_t cols);
void call_matmult(float* a, float* b, float* c, size_t a_dim, size_t b_dim, size_t n_dim);
void call_transpose(float* a, float* b, size_t a_dim, size_t b_dim);
void call_sum(float* a, float* b, size_t N);