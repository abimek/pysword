import func

class LossFunction():
    def loss(self, output, expected):
        pass

class LeastSquare(LossFunction):
    def loss(self, output, expected):
        diff = output-expected
        mul = diff*diff
        return func.sum(mul)
