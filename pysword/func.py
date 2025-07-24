from .tensor import Tensor
from .basefunc import Function
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
        return np.sum(x)

    def backward(self, output):
        t = Tensor(np.ones(self.x.value().shape))
        self.x.backward(t*output)

class ReLUFunction(OneValFunction):
    def forward(self, x):
        self.v = x * (x > 0)
        return self.v

    def backward(self, output):
        d = (1*(self.x.value()>0))
        g = Tensor(d)*output
        self.x.backward(g)

def relu(x):
    return one_var_func(x, ReLUFunction)

def sum(x):
    return one_var_func(x, SumFunction)
