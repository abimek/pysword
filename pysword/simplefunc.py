from .basefunc import Function

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
