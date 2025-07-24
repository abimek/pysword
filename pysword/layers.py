from .tensor import Tensor
from .func import relu as ReLU
import numpy as np

class Layer():
    def forward(self, data):
        pass

class LinearLayer(Layer):
    def __init__(self, output_dim):
        self.output_dim = output_dim 
        self.bias = Tensor(np.random.random((1, output_dim))*.01)

    def build_weights(self, input_dim):
        self.weights = Tensor(np.random.random((input_dim, self.output_dim))*.01)

    def forward(self, data):
        return data@self.weights + self.bias

class ReLULayer(Layer):
    def __init__(self):
        pass

    def build_weights(self, input_dim):
        self.output_dim = input_dim

    def forward(self, data):
        return ReLU(data)

