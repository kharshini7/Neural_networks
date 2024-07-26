# Kandimalla, Harshini
# 1001_960_046
# 2023_02_26
# Assignment_01_01

import numpy as np

def sigmoid(x):
    # This function calculates the sigmoid function
    # x: input
    # return: sigmoid(x)
    return 1 / (1 + np.exp(-x))


def multi_layer_nn(X_train, Y_train, X_test, Y_test, layers, alpha, epochs, h=0.00001, seed=2):
    np.random.seed(seed)
    return_train_predicts = False 

    # Number of inputs sent in each layer
    no_of_nodes = [X_train.shape[0]] + layers + [Y_train.shape[0]]

    # Initialize weight matrices for each layer
    wgts = []
    for i in range(len(no_of_nodes) - 1):
        W = np.random.randn(no_of_nodes[i + 1], no_of_nodes[i] + 1) * np.sqrt(1 / no_of_nodes[i])
        wgts.append(W)

    # creating List to store the value of error after each epoch
    error = []

    # creating List to store the predictions of train set after each epoch
    train_predicts = []

    # Train the network
    for epoch in range(epochs):
        # Forward propagation
        
        H_K = [X_train] #creates list to store activations
        Z = [] #creating list to store values of output before applying sigmoid
        i = 0
        while i < len(wgts):
            z = np.dot(wgts[i], np.vstack((np.ones((1, H_K[i].shape[1])), H_K[i])))
            Z.append(z)
            a = sigmoid(z)
            H_K.append(a)
            i += 1

        # Helps to Calculate MSE
        err = Y_train - H_K[-1]
        mse = np.mean(np.square(err))
        error.append(mse)

        # Stores  predictions of training set
        if return_train_predicts:
            train_predicts.append(H_K[-1])

        # Backward propagation
        diff = err * H_K[-1] * (1 - H_K[-1])
        h_X_K = [np.dot(diff, np.hstack((np.ones((H_K[-2].shape[1], 1)), H_K[-2].T)))]

        for i in range(len(wgts) - 1, 0, -1):
            diff = np.dot(wgts[i][:, 1:].T, diff) * H_K[i] * (1 - H_K[i])
            d = np.dot(diff, np.hstack((np.ones((H_K[i - 1].shape[1], 1)), H_K[i - 1].T)))
            h_X_K.insert(0, d)

        # Updating the  weights with new weights
        i = 0
        while i < (len(wgts)-1):
            wgts[i] += alpha * h_X_K[i]
            i += 1

    # Test the network
    h_k_test = [X_test]
    i = 0
    while i < len(wgts):
        test_h = sigmoid(np.dot(wgts[i], np.vstack((np.ones((1, h_k_test[i].shape[1])), h_k_test[i]))))
        h_k_test.append(test_h)
        i+=1

    output = h_k_test[-1]

    if return_train_predicts:
        return wgts, error, output, train_predicts
    else:
        return wgts, error, output

