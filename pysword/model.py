from .loss import LeastSquare 

class Model():
    def __init__(self, input_dim, loss=LeastSquare()):
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

    def step(self, coef):
        for layer in self.layers:
            for weight in layer.weights():
                weight.step(coef)

