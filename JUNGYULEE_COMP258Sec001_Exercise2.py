# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 17:45:59 2024

Jungyu Lee
301236221

ANN Hepatitis Classification App 
"""
import random
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

# derivative
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            train_accuracy = self.evaluate(training_data) / n
            
            if test_data:
                test_accuracy = self.evaluate(test_data) / n_test
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test));
                print("Train accuracy : {:.2f}".format(train_accuracy))
                print("Test accuracy : {:.2f}".format(test_accuracy))
            else:
                print("Epoch {} complete".format(j,))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        # print("Weights before update:")
        # for w in self.weights:
        #     print(w)

        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        
        # print("Weights after update:")
        # for w in self.weights:
        #     print(w)

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        # output layer
        # delta = (y hat - y) * derivative
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # hidden layer
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) # unlike MNIST it's not scalar
                            for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)
    
    
# load training and test data
with open('hepatitis_training_data.json') as f:
    training_data = json.load(f)

with open('hepatitis_testing_data.json') as f:
    test_data = json.load(f)

df_train = pd.DataFrame(training_data)
df_test = pd.DataFrame(test_data)

# preprocessing
X_train = df_train.drop('Die_Live', axis=1)
y_train = df_train['Die_Live']

X_test = df_test.drop('Die_Live', axis=1)
y_test = df_test['Die_Live']

y_train = [np.array([1, 0]) if label == 1 else np.array([0, 1]) for label in y_train]
y_test = [np.array([1, 0]) if label == 1 else np.array([0, 1]) for label in y_test]

# fill the missing data with mean
X_train = np.nan_to_num(X_train, nan=np.nanmean(X_train, axis=0))
X_test = np.nan_to_num(X_test, nan=np.nanmean(X_test, axis=0))

# scale the data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# network structure
input_size = X_train.shape[1]
hidden1_size = 30 # 10 
hidden2_size = 15 # 5
output_size = 2  # Two neurons for DIE and LIVE

network = Network([input_size, hidden1_size, hidden2_size, output_size])

# flattening
formatted_train_data = [(np.array([x]).reshape((input_size, 1)), np.array([y]).reshape((output_size, 1))) for x, y in zip(X_train, y_train)]

# flattening
formatted_test_data = [(np.array([x]).reshape((input_size, 1)), np.array([y]).reshape((output_size, 1))) for x, y in zip(X_test, y_test)]

# hyperparameters
epochs = 10000 # 3000
mini_batch_size = 8 # 8
learning_rate = 0.01 # 0.01

# train
network.SGD(formatted_train_data, epochs, mini_batch_size, learning_rate, test_data=formatted_test_data)

