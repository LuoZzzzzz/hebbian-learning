import numpy as np
import matplotlib.pyplot as plt
import random

from IPython.display import clear_output
import time

"""
Plots the past activities of the input neurons and neurons

Inputs:
network: NeuralNetwork object
input_neuron_history: list of np arrays containing input neurons that fired
neuron_history: list of np arrays containing neurons that fired
"""
def show_history(network, input_neuron_history, neuron_history):
    num_input_neurons = len(network.input_neurons)
    num_neurons = len(network.neurons)

    num_steps = len(neuron_history)

    input_neurons_array = np.ones((num_input_neurons, num_steps, 3))
    neurons_array = np.ones((num_neurons, num_steps, 3))

    for step in range(num_steps):
        for index in input_neuron_history[step]:
            input_neurons_array[index, step, :] = [1, 0, 0]
        
        for index in neuron_history[step]:
            neurons_array[index, step, :] = [0, 0, 1]

    plt.figure(dpi=300)
    plt.imshow(input_neurons_array)
    plt.axis("off")
    plt.show()

    plt.figure(dpi=300)
    plt.imshow(neurons_array)
    plt.axis("off")
    plt.show()


"""
Creates an animation of neurons firing.

Inputs:
network: NeuralNetwork object
input_neuron_history: list of np arrays containing input neurons that fired
neuron_history: list of np arrays containing neurons that fired
"""
def animate(network, input_neuron_history, neuron_history):
    
    input_neuron_positions = network.input_positions
    neuron_positions = network.positions

    for idx in range(len(neuron_history)):

        plt.cla() 

        # Plot neurons
        plt.scatter(input_neuron_positions[:, 0], input_neuron_positions[:, 1], s=100, c="green", edgecolor='black', zorder=3)

        # Plot neurons
        plt.scatter(neuron_positions[:, 0], neuron_positions[:, 1], s=100, c="blue", edgecolor='black', zorder=3)

        # Plot fired input neurons
        if not(input_neuron_history[idx].size == 0):
            plt.scatter(input_neuron_positions[input_neuron_history[idx]][:, 0], input_neuron_positions[input_neuron_history[idx]][:, 1], s=100, 
                        c="yellow", edgecolor='black', zorder=3)
        
        # Plot fired neurons
        if not(neuron_history[idx].size == 0):
            plt.scatter(neuron_positions[neuron_history[idx]][:, 0], neuron_positions[neuron_history[idx]][:, 1], s=100, 
                        c="red", edgecolor='black', zorder=3)
        
        # Plot weights for neurons
        for i, neuron in enumerate(network.neurons):
                for conn, _ in neuron.connections:
                    j = (network.neurons).index(conn)
                    plt.plot([neuron_positions[i, 0], neuron_positions[j, 0]],
                            [neuron_positions[i, 1], neuron_positions[j, 1]],
                            "gray", alpha=0.5, zorder=1)
                    
        # Plot weights for input neurons
        for i, neuron in enumerate(network.input_neurons):
                for conn, weight in neuron.connections:
                    j = (network.neurons).index(conn)
                    plt.plot([input_neuron_positions[i, 0], neuron_positions[j, 0]],
                            [input_neuron_positions[i, 1], neuron_positions[j, 1]],
                            "gray", alpha=weight/10, zorder=1)
        
        plt.title("Iteration " + str(idx))
        plt.axis("off")
        plt.show()

        clear_output(wait=True)
        time.sleep(0.1)