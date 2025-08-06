from pysword.tensor import Tensor 
from .basefunc import Function
from .config import get_device
import numpy as np

def one_var_func(x, opp):
    opperation = opp(x)
    if x.requires_grad:
        return Tensor(opperation.forward(x.value()), True,
                      opperation)
    return Tensor(opperation.forward(x.value()))

class OneValFunction(Function):
    def __init__(self, x):
        self.x = x

class SumFunction(OneValFunction):
    def forward(self, x):
        if get_device() == "cpu":
            return np.sum(x)

    def backward(self, output):
        if get_device() == "cpu":
            t = Tensor(np.ones(self.x.value().shape))
        self.x.backward(t*output)

class NormFunction(OneValFunction):
    def forward(self, x):
        mag = np.linalg.norm(x)
        return x * (1/mag)

    def backward(self, output):
        mag = np.linalg.norm(self.x.value())
        r = Tensor(1/mag)
        right_hand = r@r@(self.x@(self.x.transpose()@output))
        self.x.backward(r@(output-right_hand))
        

class ReLUFunction(OneValFunction):
    def forward(self, x):
        self.v = x * (x > 0)
        return self.v

    def backward(self, output):
        d = (1*(self.x.value()>0))
        g = Tensor(d)*output
        self.x.backward(g)

def norm(x):
    return one_var_func(x, NormFunction)

def relu(x):
    return one_var_func(x, ReLUFunction)

def sum(x):
    return one_var_func(x, SumFunction)
