from warnings import warn
from .simplefunc import AddFunction
from .simplefunc import SubFunction 
import numpy as np
from .simplefunc import MultFunction 
from .simplefunc import EMultFunction 

def binop_wrap(left, right, opp, leftb=True):
    main = left if leftb else right 
    other = right if leftb else left
    if isinstance(other, (float, int)):
        other = np.array([[other]])

    if isinstance(main, (float, int)):
        main = np.array([[main]])

    if not isinstance(other, Tensor):
        other = Tensor(other)

    if not isinstance(main, Tensor):
        main = Tensor(main)

    opperation = opp(main, other)
    if main.requires_grad or other.requires_grad:
        return Tensor(opperation.forward(main.value(), other.value()), True,
                      opperation)
    g =  Tensor(opperation.forward(main.value(), other.value()))
    return g

class Tensor():
    def __init__(self, val, requires_grad=False, gradfunc=None):
        self.requires_grad = requires_grad
        self.gradfunc = gradfunc 
        self.val = val
        if isinstance(val, (int, float)):
            self.val = np.array([[val]])

    def step(self, step):
        if self.requires_grad:
            raise TabError()
        self.val = self.val + -1*step*self.grad.val
        self.grad = None

    def transpose(self):
        return Tensor(self.val.T)
        
    def value(self):
        return self.val

    def forward(self):
        return self.val

    def backward(self, output=None):
        if output is None:
            output = Tensor(np.ones(self.val.shape))
        if self.gradfunc is None:
            if not hasattr(self, "grad") or self.grad is None:
                self.grad = Tensor(np.zeros(self.val.shape))
            self.grad = self.grad + output
        elif self.requires_grad:
            self.gradfunc.backward(output)

    def reset_grads(self):
        self.grad = Tensor(np.zeros(self.val.shape))

    def __add__(self, other):
        return binop_wrap(self, other, AddFunction)

    def __radd__(self, other):
        return binop_wrap(self, other, AddFunction, False)

    def __sub__(self, other):
        return binop_wrap(self, other, SubFunction)

    def __rsub__(self, other):
        return binop_wrap(self, other, SubFunction, False)

    def __mul__(self, other):
        return binop_wrap(self, other, EMultFunction)

    def __rmul__(self, other):
        return binop_wrap(self, other, EMultFunction, False)

    def __matmul__(self, other):
        return binop_wrap(self, other, MultFunction)

