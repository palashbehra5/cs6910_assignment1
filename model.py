import numpy as np
from functions import functions

class model:

  def __init__(self , params, x_train, y_train):

    self.params = params

    # L, Hidden Layers + 2, Total number of layers
    # It is suggested to have L>=1
    self.L = params['num_layers']+2

    # n
    self.n = params['hidden_size']

    # d
    self.d = params['input_layer_size']

    # k 
    self.k = params['output_layer_size']

    # N
    self.N = params['N']

    # Activation Function
    self.activation = params['activation']

    # Output Function
    self.output = params['output']

    # Loss Function
    self.loss = params['loss']

    # Weight Matrices, List of 2d matrices, each of (prev_layer_size)*(next_layer_size).
    # Bias Vectors, List of vectors, each of (next_layer_size).
    
    #  __          __    _         _      _                              _     ____   _                        
    #  \ \        / /   (_)       | |    | |                            | |   |  _ \ (_)                       
    #   \ \  /\  / /___  _   __ _ | |__  | |_  ___      __ _  _ __    __| |   | |_) | _   __ _  ___   ___  ___ 
    #    \ \/  \/ // _ \| | / _` || '_ \ | __|/ __|    / _` || '_ \  / _` |   |  _ < | | / _` |/ __| / _ \/ __|
    #     \  /\  /|  __/| || (_| || | | || |_ \__ \   | (_| || | | || (_| |   | |_) || || (_| |\__ \|  __/\__ \
    #      \/  \/  \___||_| \__, ||_| |_| \__||___/    \__,_||_| |_| \__,_|   |____/ |_| \__,_||___/ \___||___/
    #                        __/ |                                                                             
    #                       |___/   


    # Total weights,biases required : L + 1
    self.W = [[]]
    self.b = [[]]

    # Input-Hidden Layer
    self.W.append(np.random.rand(self.d,self.n))
    self.b.append(np.random.rand(self.n,))

    # Hidden-Hidden Layer
    for i in range(1,self.L-2) : self.W.append(np.random.rand(self.n,self.n))
    for i in range(1,self.L-2) : self.b.append(np.random.rand(self.n,))

    # Hidden-Output Layer
    self.W.append(np.random.rand(self.n,self.k))
    self.b.append(np.random.rand(self.k,))

    # Preactivation, Postactivation and y_hat vectors.
    # h size : (layer_size), total L+1 vectors
    # a size : (layer_size), total L+1 vectors
    # y_hat size : one single vecto, (N x k)

    self.a = [[]]
    self.h = []
    self.y_hat = np.zeros((self.k,))

    # h[0] is input layer
    self.h.append(np.zeros((self.d,)))

    # Appending vectors for L-layers
    for i in range(1,self.L-1): self.h.append(np.zeros((self.n,)))
    for i in range(1,self.L-1): self.a.append(np.zeros((self.n,)))

    # a[L] is output layer
    self.a.append(np.zeros((self.k,)))

    self.LOSS = 0

    #  ______                                     _ 
    # |  ___|                                   | |
    # | |_  ___   _ __ __      __ __ _  _ __  __| |
    # |  _|/ _ \ | '__|\ \ /\ / // _` || '__|/ _` |
    # | | | (_) || |    \ V  V /| (_| || |  | (_| |
    # \_|  \___/ |_|     \_/\_/  \__,_||_|   \__,_|
                                             
  def forward(self, X, y):

    ## Forward pass for a data point X, and a label y

    for k in range(1,self.L-1):

      print("k = {0}, a_({0}) = {2}, b_({0}) = {3}, W_({0}) = {4}, h_({1}) = {5}".format(k,k-1,self.a[k].shape,self.b[k].shape,self.W[k].shape,self.h[k-1].shape))
      self.a[k] = self.b[k] + (self.W[k].T @ self.h[k-1])
      self.h[k] = functions[self.activation](self.a[k])

    self.a[self.L-1] = self.b[self.L-1] + (self.W[self.L-1].T @ self.h[self.L-2])

    self.y_hat = functions[self.output](self.a[self.L-1])

    print(self.y_hat)

  ## Debugging Code

  def describe(self):

    print("---Architecture---\n")
    print("Input layer ({0}).".format(self.d))
    print("({0}) Hidden Layers ({1}).".format(self.L,self.n))
    print("Activation Function ({0}).".format(self.activation))
    print("Output layer ({0}).".format(self.k))
    print("Output Function ({0}).".format(self.output))
    print("Loss Function ({0}).\n".format(self.loss))

    print("---WandB---\n")
    print("Weights : ")
    for i in range(1,len(self.W)) :  print(self.W[i].shape)
    print("Biases : ")
    for i in range(1,len(self.b)) : print(self.b[i].shape)
    print("\n")

    print("---Activation-Layers---\n")
    print("pre-activation : ")
    for i in range(1,len(self.a)) : print(self.a[i].shape)
    print("Activation : ")
    for i in self.h : print(i.shape)
    print("\n")