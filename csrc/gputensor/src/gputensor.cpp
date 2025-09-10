#include "gputensor.h"
#include <memory>
#include <algorithm>
#include <stdexcept>
#include "ops.h"

size_t Shape::size() {
  return r * c;
}

bool Shape::operator==(Shape s) const {
  return s.r == r && s.c == c;
}

bool Shape::operator!=(Shape s) const {
  return s.r != r || s.c != c;
}

void CudaDeleter::operator()(float* d) {
  cudaFree(d);
}

StorageLocation Storage::get_location() {
  return location;
}

float* Storage::gpu() {
  return gpu_data.get();
}

size_t Storage::get_size() {
  return size;
}

float* Storage::cpu() {
  return cpu_data.get();
}

void Storage::try_move_to_cpu() {
  if (location == StorageLocation::GPU) {
    location = StorageLocation::CPU;
    cpu_data = std::make_unique<float[]>(size);
    cudaMemcpy(cpu(), gpu(), size * sizeof(float), cudaMemcpyDeviceToHost);
    gpu_data.reset();
  }
}

void Storage::try_move_to_gpu() {
  if (location == StorageLocation::CPU) {
    location = StorageLocation::GPU;
    float* gpdata;
    cudaMalloc(&gpdata, size * sizeof(float));
    gpu_data = std::unique_ptr<float, CudaDeleter>(gpdata);
    cudaMemcpy(gpu(), cpu(), size * sizeof(float), cudaMemcpyHostToDevice);
    cpu_data.reset();
  }
}

Storage::Storage(size_t size) 
  : size(size)
  , cpu_data(std::make_unique<float[]>(size)) {
    gpu_data = std::unique_ptr<float, CudaDeleter>();
}

/*
--------------------------------------------------------------------------------
*/

GPUTensor::GPUTensor(Shape ishape) : storage(Storage(ishape.size())) {
  shape = ishape;
}

float* GPUTensor::data() {
  storage.try_move_to_cpu();
  return storage.cpu();
}

/*
* Params could prob be improved to use a vector for initial
*/
void GPUTensor::populate_data(float* input, size_t length) {
  std::copy(input, input+length, data());
}

Shape GPUTensor::get_shape() {
  return shape;
} 

GPUTensor GPUTensor::mult(float value) {
  GPUTensor newTensor = GPUTensor(shape);
  newTensor.storage.try_move_to_gpu();
  storage.try_move_to_gpu();
  call_k_mult(storage.gpu(), newTensor.storage.gpu(), shape.r, shape.c, value);
  return newTensor;
}

GPUTensor GPUTensor::add(GPUTensor& tensor) {
  if (tensor.shape != shape) {
    throw std::runtime_error("Mismatch shapes");
  }
  GPUTensor newTensor = GPUTensor(shape);
  newTensor.storage.try_move_to_gpu();
  tensor.storage.try_move_to_gpu();
  storage.try_move_to_gpu();
  call_p_add(storage.gpu(), tensor.storage.gpu(), newTensor.storage.gpu(), shape.r, shape.c);
  return newTensor;
}

float GPUTensor::sum() {
  if (shape.c != 1 && shape.r != 1) {
    throw std::runtime_error("Expected 1d vector");
  }
  Storage store(1);
  store.try_move_to_gpu();
  storage.try_move_to_gpu();
  call_sum(storage.gpu(), store.gpu(), storage.get_size());
  store.try_move_to_cpu();
  return *(store.cpu());
}

GPUTensor GPUTensor::mult(GPUTensor& tensor) {
  if (tensor.shape != shape) {
    throw std::runtime_error("Mismatch shapes");
  }
  GPUTensor newTensor = GPUTensor(shape);
  newTensor.storage.try_move_to_gpu();
  tensor.storage.try_move_to_gpu();
  storage.try_move_to_gpu();
  call_p_mult(storage.gpu(), tensor.storage.gpu(), newTensor.storage.gpu(), shape.r, shape.c);
  return newTensor;
}

GPUTensor GPUTensor::tranpose() {
  GPUTensor newTensor = GPUTensor(Shape{shape.c, shape.r});
  newTensor.storage.try_move_to_gpu();
  storage.try_move_to_gpu();
  call_transpose(storage.gpu(), newTensor.storage.gpu(), shape.r, shape.c);
  return newTensor;
}

GPUTensor GPUTensor::matmul(GPUTensor& tensor) {
  if (shape.c != tensor.shape.r) {
    throw std::runtime_error("Mismatch shapes");
  }
  GPUTensor newTensor = GPUTensor(Shape{shape.r, tensor.shape.c});
  newTensor.storage.try_move_to_gpu();
  tensor.storage.try_move_to_gpu();
  storage.try_move_to_gpu();
  call_matmult(storage.gpu(), tensor.storage.gpu(), newTensor.storage.gpu(), shape.r, tensor.shape.c, shape.c);
  return newTensor;
}
