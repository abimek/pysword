from warnings import warn
import numpy as np
from tensor import Tensor
import func
import loss

class Model():
    def __init__(self, input_dim, loss=loss.LeastSquare()):
        self.input_dim = input_dim
        self.layers = []
        self.loss = loss

    def add_layer(self, layer):
        if len(self.layers) == 0:
            layer.build_weights(self.input_dim)
        else:
            layer.build_weights(self.layers[len(self.layers)-1].output_dim)
        self.layers.append(layer)

    def forward(self, data):
        for layer in self.layers:
            data = layer.forward(data)
        return data

    def training_loss(self, data, expected):
        data.requires_grad = True
        d = self.forward(data)
        return d, self.loss.loss(d, expected)
    
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
        return func.relu(data)

model = Model(input_dim=5)
model.add_layer(LinearLayer(output_dim=3))
model.add_layer(ReLULayer())
model.add_layer(LinearLayer(output_dim=100))
model.add_layer(ReLULayer())
model.add_layer(LinearLayer(output_dim=100))
model.add_layer(ReLULayer())
model.add_layer(LinearLayer(output_dim=3))

step = 0.01

for x in range(2000):
    actual, g = model.training_loss(Tensor(np.array([[3, 2, 90, 3, 3]])), Tensor(np.array([[1, 5, 1]])))
    if x % 100 == 0:
        print("actual: ", actual.value())
        print("loss: ", g.value())
    g.backward()
    model.layers[0].weights.step(step)
    model.layers[0].bias.step(step)
    model.layers[2].weights.step(step)
    model.layers[2].bias.step(step)
    model.layers[4].weights.step(step)
    model.layers[4].bias.step(step)
    model.layers[6].weights.step(step)
    model.layers[6].bias.step(step)
