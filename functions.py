import numpy as np

def sigmoid(z):

    sigmoid = 1/(1+(np.exp(-z)))
    sigmoid[sigmoid==0] = 1e-4
    sigmoid[sigmoid==1] = 1-1e-4
    return sigmoid

def d_sigmoid(z):

    d_sigmoid = sigmoid(z)*(1-sigmoid(z))
    d_sigmoid[d_sigmoid==0] = 1e-4
    d_sigmoid[d_sigmoid==1] = 1-1e-4
    return d_sigmoid

def softmax(a_l):

    # Scaling 
    e_x = np.exp(a_l - np.max(a_l))
    return e_x / e_x.sum()

def relu(z):
    
    z[z<0]=0
    return z

def d_relu(z):
    
    z[z<=0]=0
    z[z>0]=1
    return z

def tanh(z):

    z = np.tanh(z)
    z[z==1] = 1-1e-4
    z[z==-1] = -(1-1e-4)
    return z

def d_tanh(z):

    return 1-(tanh(z)**2)

def e_l(y,k):

    e_l = np.zeros((k,1))
    e_l[y][0] = 1;

    return e_l


functions = {

    "sigmoid":sigmoid,
    "d_sigmoid":d_sigmoid,
    "softmax":softmax,
    "relu":relu,
    "d_relu":d_relu,
    "tanh":tanh,
    "d_tanh":d_tanh,
    "e_l":e_l

}