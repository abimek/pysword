#pragma once
#include <memory>

/**
 * 
 * from GpuTensor import GpuTensor 
 * 
 * myArray = GpuTensor(np.ndarray([[....]]))
 * 
 * myArray @ myArray -> newGpu Tensor
 * 
 * numpyObject = myArray.to_numpy()
 */

enum StorageLocation {GPU, CPU};

struct Shape {
  int r;
  int c;

  size_t size();
  bool operator==(Shape s) const;
  bool operator!=(Shape s) const;
};

struct CudaDeleter {
    void operator()(float* d);
};

class Storage {
public:
  StorageLocation location{StorageLocation::CPU};
  Storage(size_t size);
  StorageLocation get_location();
  size_t get_size();
  float* gpu();
  float* cpu();
  void try_move_to_gpu();
  void try_move_to_cpu();

private:
    size_t size;
    std::unique_ptr<float[]> cpu_data;
    std::unique_ptr<float, CudaDeleter> gpu_data;
};

class GPUTensor {
public:
    GPUTensor(Shape ishape);
    GPUTensor(Shape shape, Storage&& storage) : shape(shape), storage(std::move(storage)) {};

    Shape get_shape();

    float sum();
    GPUTensor mult(float value);
    GPUTensor mult(GPUTensor& tensor);
    GPUTensor matmul(GPUTensor& tensor);
    GPUTensor tranpose();
    GPUTensor add(GPUTensor& tensor);
    /**
     * May cause an issue if the size of the underlying data array
     * isn't of the correct size
     * 
     */
    void populate_data(float* input, size_t length);

    // returns pointer to the underlying data on the cpu, gets swapped
    // over to the cpu if called and currently on the gpu.
    float* data();
    //float* to_numpy(); <- implement when needed
private:
  Shape shape;
  Storage storage;
};

