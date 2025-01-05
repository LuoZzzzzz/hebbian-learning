import numpy as np
import matplotlib.pyplot as plt
import random


"""
Neuron class

Attributes:
"""
class Neuron:
    def __init__(self, threshold=1.0, is_input=False, hebb=0.0):
        self.threshold = threshold
        self.potential = 0.0
        self.connections = []  # List of tuples (Neuron, weight)
        self.is_input = is_input  # Flag for input neurons
        self.hebb = hebb # Hebbian learning rate

    def connect(self, neuron, weight=1.0):
        self.connections.append((neuron, weight))

    def stimulate(self, input_potential):
        self.potential += input_potential

    def fire(self):
        if self.potential >= self.threshold:
            for idx, (neuron, weight) in enumerate(self.connections):
                neuron.stimulate(input_potential=weight)
                if neuron.potential >= neuron.threshold:  # Target neuron fired ####NEW
                    self.connections[idx] = (neuron, weight + self.hebb)  # Increment weight ####NEW
            self.potential = 0.0  # Reset potential after firing
            return True  # Indicate that the neuron fired
        return False
    

"""
NeuralNetwork class

Attributes:
"""      
class NeuralNetwork:
    def __init__(self, size=10, input_size=2, threshold=1.0):
        self.neurons = [Neuron(threshold) for _ in range(size)]
        self.input_neurons = [Neuron(threshold, is_input=True) for _ in range(input_size)]
        self.positions = np.random.rand(size, 2)
        self.input_positions = np.random.rand(input_size, 2)

    def connect_randomly(self, connection_probability=0.3, weight_range=(0.5, 1.5)):
        for neuron in self.neurons + self.input_neurons:
            for other_neuron in self.neurons:
                if neuron != other_neuron and random.random() < connection_probability:
                    weight = random.uniform(*weight_range)
                    neuron.connect(other_neuron, weight)

    def stimulate_inputs(self, step):
        for input_neuron in self.input_neurons:
            if (step % 10 == 0) and (step <= 300):  # Periodic firing based on index
                input_neuron.stimulate(input_potential=1.0)

    def step(self, step):
        fired_neurons = []
        fired_input_neurons = []

        self.stimulate_inputs(step)

        for neuron in self.input_neurons + self.neurons:
            if neuron.fire():
                if neuron.is_input:
                    fired_input_neurons.append(neuron)
                else:
                    fired_neurons.append(neuron)

        # Return indices of fired input neurons and fired neurons
        return np.asarray([self.input_neurons.index(neuron) for neuron in fired_input_neurons]), np.asarray([self.neurons.index(neuron) for neuron in fired_neurons])
    
    def simulate(self, iterations):
        input_neuron_history = []
        neuron_history = []
        for idx in range(iterations):
            input_neuron_indices, neuron_indices = self.step(step=idx)
            input_neuron_history.append(input_neuron_indices)
            neuron_history.append(neuron_indices)
        return input_neuron_history, neuron_history