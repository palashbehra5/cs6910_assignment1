import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functions import e_l,functions

class model:

  def __init__(self , params):

    # Initializing Model Parameters
    # n : Hidden layer size in Neurons
    # d : Input layer size in Neurons
    # k : Output layer size in Neurons
    # N : Number of datapoints                                                                                
    # L, Hidden Layers + 2, Total number of layers
    # It is suggested to have at least one hidden layer
    self.L = params['num_layers']+2
    self.n = params['hidden_size']
    self.d = params['input_layer_size']
    self.k = params['output_layer_size']
    self.N = params['N']
    self.activation = params['activation']
    self.output = params['output']
    self.loss = params['loss']
    self.weight_init = params['weight_init']
    # Weight Matrices, List of 2d matrices, each of (prev_layer_size)*(next_layer_size).
    # Bias Vectors, List of vectors, each of (next_layer_size).
    # Creating local variables

    n = self.n
    d = self.d
    L = self.L
    k = self.k

    # Initializing Weights and Biases
    # Total weights,biases required : L-1
    self.W = [[]]
    self.b = [[]]

    if(self.weight_init=="random") : 

      # Input-Hidden Layer
      self.W.append(np.random.uniform(-1,1,[n,d]))
      self.b.append(np.random.uniform(-1,1,[n,1]))

      # Hidden-Hidden Layer
      for i in range(1,L-2) : self.W.append(np.random.uniform(-1,1,[n,n]))
      for i in range(1,L-2) : self.b.append(np.random.uniform(-1,1,[n,1]))

      # Hidden-Output Layer
      self.W.append(np.random.uniform(-1,1,[k,n]))
      self.b.append(np.random.uniform(-1,1,[k,1]))

    elif(self.weight_init=="xavier") : 

      # Input-Hidden Layer
      self.W.append(np.random.uniform(-(6/(n+d))**0.5,(6/(n+d))**0.5,[n,d]))
      self.b.append(np.random.uniform(-(6/(n+1))**0.5,(6/(n+1))**0.5,[n,1]))

      # Hidden-Hidden Layer
      for i in range(1,L-2) : self.W.append(np.random.uniform(-(6/(n+n))**0.5,(6/(n+n))**0.5,[n,n]))
      for i in range(1,L-2) : self.b.append(np.random.uniform(-(6/(n+1))**0.5,(6/(n+1))**0.5,[n,1]))

      # Hidden-Output Layer
      self.W.append(np.random.uniform(-(6/(k+n))**0.5,(6/(k+n))**0.5,[k,n]))
      self.b.append(np.random.uniform(-(6/(k+1))**0.5,(6/(k+1))**0.5,[k,1]))

    # Preactivation, Postactivation and y_hat vectors.
    # h size : (layer_size), total L+1 vectors
    # a size : (layer_size), total L+1 vectors
    # y_hat size : one single vecto, (N x k)
    # Feed Forwward function (model_instance, X[i]) : 
                                                                                                                                                                                        
  def forward(self, X):

    L = self.L
    a = [0]*L
    h = [0]*L

    # Setting Input Layer
    h[0] = X.reshape(len(X),1)

    # Forward pass for a data point X, and a label y
    for k in range(1,L-1):

      a[k] = self.b[k] + (self.W[k] @ h[k-1])
      h[k] = functions[self.activation](a[k])

    a[L-1] = self.b[L-1] + (self.W[L-1] @ h[L-2])
    y_hat = functions[self.output](a[L-1])

    return {'y_hat':y_hat,'a':a,'h':h}

  def backpropagate(self,y,params):

    y_hat = params['y_hat']
    a = params['a']
    h = params['h']
    L = self.L
    k = self.k

    # Note : dw[0] and db[0] are both dummy variables
    dw_ = [0]*(L)
    db_ = [0]*(L)
    
    # Gradients of Loss funtion WRT output layer preactivation
    da = functions["d_"+self.loss](y,k,y_hat)

    for i in range(L-1,0,-1):

      dw_[i] =  da @ h[i-1].T
      db_[i] = da

      if(i==1) : break

      dh = self.W[i].T @ da
      da = dh * functions["d_"+self.activation](a[i-1])
      
    # Returned values to be accumulated
    return {'dw_':dw_,'db_':db_}
