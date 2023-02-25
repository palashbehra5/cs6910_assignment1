import numpy as np

def sigmoid(z):

    # Scaling 
    z = (z - min(z) )/ (max(z) - min(z))
    return 1/(1+(np.e**(-z)))

def d_sigmoid(z):

    # Scaling 
    z = (z - min(z) )/ (max(z) - min(z))
    return sigmoid(z)*(1-sigmoid(z))

def softmax(a_l):

    # Scaling 
    a_l = (a_l - min(a_l) )/ (max(a_l) - min(a_l))
    return (np.e**a_l)/np.sum(np.e**(a_l))

def relu(z):
    
    z[z<0]=0
    return z

def d_relu(z):
    
    z[z<=0]=0
    z[z>0]=1
    return z

def tanh(z):

    # Scaling 
    z = (z - min(z) )/ (max(z) - min(z))
    return np.tanh(z)

def d_tanh(z):

    # Scaling 
    z = (z - min(z) )/ (max(z) - min(z))
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