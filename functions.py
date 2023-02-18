import numpy as np

def sigmoid(z):

    return 1/(1+(np.e**(-z)))

def d_sigmoid(z):

    return sigmoid(z)*(1-sigmoid(z))

def softmax(a_l):

    # Scaling features
    a_l = (a_l - min(a_l) )/ (max(a_l) - min(a_l))

    return (np.e**a_l)/np.sum(np.e**(a_l))

def relu(z):

    return max(0,z)

def d_relu(z):

    return 0 if z<=0 else 1

def tanh(z):

    return np.tanh(z)

def d_tanh(z):

    return 1-(tanh(z)**2)


functions = {

    "sigmoid":sigmoid,
    "d_sigmoid":d_sigmoid,
    "softmax":softmax,
    "relu":relu,
    "d_relu":d_relu,
    "tanh":tanh,
    "d_tanh":d_tanh

}