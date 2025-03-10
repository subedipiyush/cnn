import random

from my_types import Vector, FloatVector
from activation_functions import ActivationFn, ThresholdFn
from kernel_initializers import KernelInitializer, RandomNormal

class Layer:
    def __init__(self, activation_fn: ActivationFn) -> None:
        self.activation_fn = activation_fn

    def feed_forward(self, input: Vector) -> Vector:
        assert NotImplemented

class DenseLayer(Layer):
    def __init__(self, num_inputs: int, num_nodes: int, activation_fn: ActivationFn, kernel_initializer: KernelInitializer) -> None:
        super().__init__(activation_fn)
        self.weights = kernel_initializer.initialize((num_nodes, num_inputs))
        self.biases = kernel_initializer.initialize((1, num_nodes))

    def feed_forward(self, input: Vector) -> Vector:
        z = self.weights.dot(input) + self.biases
        return self.activation_fn.apply(z)

    def __str__(self) -> str:
        return f"DenseLayer(weights_shape={self.weights.shape}, biases_shape={self.biases.shape}, activation_fn={self.activation_fn.__class__.__name__})"

class NeuralNetwork:
    def __init__(self) -> None:
        self.hidden_layers: list[Layer] = []
    
    def add_layer(self, layer: Layer) -> None:
        self.hidden_layers.append(layer)
    
    def make_prediction(self, input: Vector) -> Vector:
        output = input
        for layer in self.hidden_layers:
            output = layer.feed_forward(output)
        return output
    
    def __str__(self) -> str:
        summary_str = f"NeuralNetwork(num_layers={len(self.hidden_layers)})\n"
        for i, layer in enumerate(self.hidden_layers):
            summary_str += f"Layer {i+1}: {layer}\n"
        return summary_str

def binary_classifier(num_input_dims: int) -> NeuralNetwork:
    nn = NeuralNetwork()
    nn.add_layer(DenseLayer(
        num_inputs=num_input_dims, num_nodes=1, 
        activation_fn=ThresholdFn(threshold=0),
        kernel_initializer=RandomNormal()))
    return nn

def main() -> None:
    b = binary_classifier(num_input_dims=3)
    print(b)
    print(b.make_prediction(FloatVector([0.1, 0.2, 0.3]).T))

if __name__ == "__main__":
    main()