import gputensor as m
import numpy as np
from pysword import model
from pysword import tensor
from pysword import layers
from gputensor import GPUTensor
import numpy as np
from gputensor import GPUTensor

def test_1():
    l1 = np.array([[1,2,3,4,5,6,7,8,9,10]], dtype=np.float32)
    g1 = GPUTensor(l1.shape, l1)
    print(g1.sum())

def test_2():

    mymod = model.Model(input_dim=5)
    mymod.add_layer(layers.LinearLayer(output_dim=3))
    mymod.add_layer(layers.NormLayer())
    mymod.add_layer(layers.ReLULayer())
    mymod.add_layer(layers.LinearLayer(output_dim=10))
    mymod.add_layer(layers.NormLayer())
    mymod.add_layer(layers.ReLULayer())
    mymod.add_layer(layers.LinearLayer(output_dim=20))
    mymod.add_layer(layers.NormLayer())
    mymod.add_layer(layers.ReLULayer())
    mymod.add_layer(layers.LinearLayer(output_dim=10))
    mymod.add_layer(layers.NormLayer())
    mymod.add_layer(layers.ReLULayer())
    mymod.add_layer(layers.LinearLayer(output_dim=3))
    step = 0.01
    for x in range(200):
        actual, g = mymod.training_loss(tensor.Tensor(np.array([[3, 2, 90, 3, 3]])),
                                        tensor.Tensor(np.array([[1, 5, 1]])))
        if x % 5 == 0:
            print("actual: ", actual.value())
            print("loss: ", g.value())
        g.backward()
        mymod.step(step)