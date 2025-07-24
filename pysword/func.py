import tensor
import numpy as np

def one_var_func(x, opp):
    opperation = opp(x)
    if x.requires_grad:
        return tensor.Tensor(opperation.forward(x.value()), True,
                      opperation)
    return tensor.Tensor(opperation.forward(x.value()))

class Function():
    def forward(self):
        pass

    def backward(self):
        pass

class BinFunction(Function):
    def __init__(self, left, right):
        self.left = left
        self.right = right

class AddFunction(BinFunction):
    def forward(self, left, right):
        return left + right
    
    def backward(self, output):
        self.left.backward(output)
        self.right.backward(output)

class OneValFunction(Function):
    def __init__(self, x):
        self.x = x

def relu(x):
    return one_var_func(x, ReLUFunction)

def sum(x):
    return one_var_func(x, SumFunction)

class SumFunction(OneValFunction):
    def forward(self, x):
        return np.sum(x)

    def backward(self, output):
        t = tensor.Tensor(np.ones(self.x.value().shape))
        self.x.backward(t*output)

class ReLUFunction(OneValFunction):
    def forward(self, x):
        self.v = x * (x > 0)
        return self.v

    def backward(self, output):
        d = (1*(self.x.value()>0))
        g = tensor.Tensor(d)*output
        self.x.backward(g)

class SubFunction(BinFunction):
    def forward(self, left, right):
        return left - right

    def backward(self, output):
        self.left.backward(output)
        self.right.backward(-1*output)

class MultFunction(BinFunction):
    def forward(self, left, right):
        return left @ right

    def backward(self, output):
        self.left.backward(output@self.right.transpose())
        self.right.backward(self.left.transpose()@output)

class EMultFunction(BinFunction):
    def forward(self, left, right):
        return left * right

    def backward(self, output):
        self.left.backward(output*self.right)
        self.right.backward(output*self.left)
