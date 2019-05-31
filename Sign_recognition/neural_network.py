"""creation of the neural network and its side functions"""

import numpy as np
import math

# CONSTANTES
alpha = 10
# si il est trop lent changer jusqu'a ce qu'a diverge
epoch_number = 10000


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(np.zeros(x.shape), x)


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu_derivative(x):
    m = x.shape
    A = np.zeros(m)
    return np.maximum(A, x)


def derivative(fct):
    if fct == sigmoid:
        return sigmoid_derivative
    elif fct == relu:
        return relu_derivative
    else:
        return "Error"


def cost(x, y):
    return -(y * np.log(x) + (1 - y) * np.log(1 - x))


def cost_basique(x, y):
    return (x - y) ** 2


def cost_derivative_basique(x, y):
    return 2 * (x - y)


def cost_derivative(x, y):
    return (x - y) / (x * (1 - x))


class Nnetwork:

    def __init__(self, layers, functions):
        self.layers = layers  # the first layer must have as many neurons as inputs and the last layer must have as many neuron as classifications
        self.values = list(np.zeros([layers[i], 1]) for i in range(len(layers)))
        self.functions = functions
        self.activations = [np.zeros([layers[0], 1])] + list(
            self.functions[i](np.zeros([layers[i + 1], 1])) for i in range(len(layers) - 1))
        self.derivatives = list(derivative(self.functions[i]) for i in range(len(layers) - 1))
        self.weights = list(np.random.rand(layers[i], layers[i - 1]) * 0.01 for i in range(1, len(layers)))
        self.bias = list(np.zeros([layers[i], 1]) for i in range(1, len(layers)))

    def calculate(self, input_vector):
        """calculates the total cost function at the output of the NN for one image input"""

        N = len(self.layers)

        a = input_vector
        for i in range(1, N):
            w = self.weights[i]
            b = self.bias[i]
            z = np.dot(w, a) + b
            a = self.functions[i](z)

        return a

    def forward_propagation(self, input_matrix, experimental_output):
        """calculates the total cost function at the output of the NN for as many images
        as you want as input"""

        N = len(self.layers)
        m = np.shape(input_matrix)
        A = input_matrix
        self.activations[0] = A
        for i in range(1, N):
            W = self.weights[i - 1]

            B = np.repeat(self.bias[i - 1], m[1], axis=1)

            Z = np.dot(W, A) + B
            self.values[i] = Z
            A = self.functions[i - 1](Z)
            self.activations[i] = A

        J = 1 / m[1] * np.sum(cost(A, experimental_output))
        return A, J

    def accuracy(self, input, output):
        score = 0
        A = self.forward_propagation(input,output)[0]
        for i in range(input.shape[1]):
            a,b,c = output[0, i], output[1,i], output[2,i]
            x,y,z = A[0, i], A[1, i], A[2, i]
            proximity = max(abs(a-x),abs(b-y),abs(z-c))
            if proximity <= 0.5:
                score += 1
        return score / A.shape[1] * 100


    def backward_propagation(self, input_matrix, output_matrix, experimental_output):
        N = len(self.layers)
        m = np.shape(input_matrix)
        dA = (1 / m[1]) * cost_derivative(output_matrix, experimental_output)

        for i in reversed(range(1, N)):
            dZ = dA * self.derivatives[i - 1](self.values[i])

            if i > 1:
                dW = (1 / m[1]) * np.dot(dZ, (self.activations[i - 1]).T)

            else:
                dW = (1 / m[1]) * np.dot(dZ, (input_matrix).T)

            dB = (1 / m[1]) * np.sum(dZ, axis=1, keepdims=True)
            dA = np.dot(self.weights[i - 1].T, dZ)

            self.weights[i - 1] = self.weights[i - 1] - alpha * dW
            self.bias[i - 1] = self.bias[i - 1] - alpha * dB

    def epoch(self, input_matrix, experimental_output):
        J = math.inf
        i = 0
        while J > 0.01:
            (A, J) = self.forward_propagation(input_matrix, experimental_output)
            if i % 50 == 0:
                print("Epoch " + str(i) + ": Cost -> " + str(J))
            self.backward_propagation(input_matrix, A, experimental_output)
            i += 1

    def epoch1(self, input_matrix, experimental_output, epoch_number2=epoch_number):
        """input_matrix de taille (nombre de pixels, nombre d images)
            experimental_output de taille ()
        """
        for i in range(epoch_number2):

            (A, J) = self.forward_propagation(input_matrix, experimental_output)

            self.backward_propagation(input_matrix, A, experimental_output)
            if i % 50 == 0:
                print("Epoch " + str(i) + ": Cost -> " + str(J))


    def epoch_with_dev_set(self, input_matrix, experimental_output, dev_input, dev_output):
        """takes the input and output and the matrix of the dev set ans trains it until
        the calculations get to a stable minimum of the dev training"""
        i = 0
        results = [2,3,4,5]
        while results != sorted(results)[::-1] or results[-1]<99:
            Acc = self.accuracy(dev_input,dev_output)
            (A, J) = self.forward_propagation(input_matrix, experimental_output)

            if i % 50 == 0:
                print("Epoch " + str(i) + ": Cost -> " + str(J))
                print("This NN has an accuracy of " + str(Acc) + "%")
            self.backward_propagation(input_matrix, A, experimental_output)
            i += 1
            results.append(Acc)
            x = results.pop(0)
            # print(results)

    def save(self):
        np.save('saved_weights.npy', self.weights)
        np.save('saved_bias.npy', self.bias)
        pass

    def load(self, weights, bias):
        self.weights = np.load(weights)
        self.bias = np.load(bias)
        pass


# test forward_propagation
# NN1 = Nnetwork([3,2,1],[sigmoid,sigmoid])
# NN1.weights=[np.array([[0.25,0.5,0.3],[0.4,0.1,0.8]]),np.array([[0.2,0.7]])]
# print(NN1.forward_propagation(np.array([[0.7,0.2],[0.8,0.1],[0.9,0.3]]),np.array([[1,0]])))
# print(NN1.weights)
# print(NN1.bias)
# print(NN1.activations)
"""result

A,J = (array([[0.65981036, 0.62637652]]), 0.700154766241492)
W = [array([[0.25, 0.5 , 0.3 ],
       [0.4 , 0.1 , 0.8 ]]), array([[0.2, 0.7]])]
b = [array([[0.],
       [0.]]), array([[0.]])]
all A = [array([[0.],
       [0.],
       [0.]]), array([[0.69951723, 0.54735762],
       [0.74649398, 0.58175938]]), array([[0.65981036, 0.62637652]])]
"""

# test back_propagation
# NN1 = Nnetwork([3,2,1],[relu,sigmoid])
# NN1.weights=[np.array([[0.25,0.5,0.3],[0.4,0.1,0.8]]),np.array([[0.2,0.7]])]
# NN1.epoch1(np.array([[0.7,0.2,0.1],[0.8,0.1,0.2],[0.9,0.3,0.4]]),np.array([[1,0,0]]),epoch_number2=600)
# print()1
# result
# [[0.5]
#  [0.5]]
# [[0.5 0.5]]
# [[0]]
# [[0.5]]
# [[0.62245933]]
# 0.47407698418010663
# [[-1.60653066]]
# [[-0.18877033 -0.18877033]]
# [[-0.37754067]]
# [[-0.37754067]]
# [[-0.18877033]
#  [-0.18877033]]
# [[0.68877033 0.68877033]]
# [[0.37754067]]


# test epoch
# NN1 = Nnetwork([3, 2, 1], [sigmoid, sigmoid])
# print(NN1.weights)
# print(NN1.bias)
# print(NN1.activations)
# NN1.epoch1(np.array([[0.7, 0.2], [0.8, 0.1], [0.9, 0.3]]), np.array([[1, 0]]))
