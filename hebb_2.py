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
        self.current_potential = 0.0
        self.next_potential = 0.0
        self.connections = []  # List of tuples (Neuron, weight)
        self.is_input = is_input  # Flag for input neurons
        self.hebb = hebb # Hebbian learning rate

    def connect(self, neuron, weight=1.0):
        self.connections.append((neuron, weight))

    # For input neurons
    def stimulate(self, input_potential):
        self.next_potential += input_potential

    def fire(self):
        if self.current_potential >= self.threshold:
            for idx, (neuron, weight) in enumerate(self.connections):
                neuron.stimulate(input_potential=weight) # update the state on the next time step

                # Check if the target neuron fires on the next iteration, and if so, strengthen connection
                if neuron.next_potential >= neuron.threshold:  # Target neuron fired 
                    self.connections[idx] = (neuron, weight + self.hebb)  # Increment weight 

            self.next_potential = 0.0  # Reset potential after firing
            return True  # Indicate that the neuron fired
        return False
    
    def update(self):
        self.current_potential = self.next_potential
    

"""
NeuralNetwork class

Attributes:
"""      
class NeuralNetwork:
    def __init__(self, size=10, input_size=2, threshold=1.0, decay_factor=1.0):
        self.neurons = [Neuron(threshold) for _ in range(size)]
        self.input_neurons = [Neuron(threshold, is_input=True) for _ in range(input_size)]
        self.positions = np.vstack((np.cos(np.linspace(start=0, stop=2*np.pi, num=size, endpoint=False)), \
                                    np.sin(np.linspace(start=0, stop=2*np.pi, num=size, endpoint=False)) + 1.2)).T
        self.input_positions = np.vstack((np.linspace(start=-0.5, stop=0.5, num=input_size, endpoint=True), \
                                           np.zeros(input_size))).T
        self.decay_factor = decay_factor

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

    def weight_decay(self):
        for neuron in self.neurons + self.input_neurons:    # Iterate through all neurons   
            for idx in range(len(neuron.connections)):      # Iterate through neuron connections and decay weights
                neuron.connections[idx] = (neuron.connections[idx][0], self.decay_factor*neuron.connections[idx][1])

    def step(self, step):
        fired_neurons = []
        fired_input_neurons = []

        self.stimulate_inputs(step)

        # Record fired neurons
        for neuron in self.input_neurons + self.neurons:
            if neuron.fire():
                if neuron.is_input:
                    fired_input_neurons.append(neuron)
                else:
                    fired_neurons.append(neuron)

        # Set current potential to next potential
        for neuron in self.input_neurons + self.neurons:
            neuron.update()

        # Decay all weights
        self.weight_decay()

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