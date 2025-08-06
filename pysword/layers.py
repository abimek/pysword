from pysword.tensor import Tensor 
from .func import relu as ReLU
from .func import norm as Norm 
from .config import get_device
import numpy as np

class Layer():
    def forward(self, data):
        pass

    def build_weights(self, input_dim):
        pass

    def weights(self):
        pass

class LinearLayer(Layer):
    def __init__(self, output_dim):
        self.output_dim = output_dim 
        self.bias = Tensor(np.random.random((1, output_dim))*.01)

    def build_weights(self, input_dim):
        if get_device() == "cpu":
            self.lin = Tensor(np.random.random((input_dim, self.output_dim))*.01)

    def weights(self):
        return (self.bias, self.lin)

    def forward(self, data):
        return data@self.lin + self.bias

class ReLULayer(Layer):
    def __init__(self):
        pass

    def build_weights(self, input_dim):
        self.output_dim = input_dim

    def weights(self):
        return tuple()

    def forward(self, data):
        return ReLU(data)

class NormLayer(Layer):
    def __init__(self):
        pass

    def build_weights(self, input_dim):
        self.output_dim = input_dim

    def weights(self):
        return tuple()

    def forward(self, data):
        return Norm(data)


