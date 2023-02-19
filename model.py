import numpy as np
from functions import functions

class model:

  def __init__(self , params):

    #  __  __             _        _      _____                                     _                   
    # |  \/  |           | |      | |    |  __ \                                   | |                  
    # | \  / |  ___    __| |  ___ | |    | |__) |__ _  _ __  __ _  _ __ ___    ___ | |_  ___  _ __  ___ 
    # | |\/| | / _ \  / _` | / _ \| |    |  ___// _` || '__|/ _` || '_ ` _ \  / _ \| __|/ _ \| '__|/ __|
    # | |  | || (_) || (_| ||  __/| |    | |   | (_| || |  | (_| || | | | | ||  __/| |_|  __/| |   \__ \
    # |_|  |_| \___/  \__,_| \___||_|    |_|    \__,_||_|   \__,_||_| |_| |_| \___| \__|\___||_|   |___/
                                                                                                   
    # Initializing Model Parameters
    # n : Hidden layer size in Neurons
    # d : Input layer size in Neurons
    # k : Output layer size in Neurons
    # N : Number of datapoints                                                                                
    # L, Hidden Layers + 2, Total number of layers
    # It is suggested to have at least one hidden layer
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

    # Creating local variables

    n = self.n
    d = self.d
    L = self.L
    k = self.k

    # Initializing Weights and Biases
    # Total weights,biases required : L + 1
    self.W = [[]]
    self.b = [[]]
    # self.dw = [[]]
    # self.db = [[]]

    # Input-Hidden Layer
    self.W.append(np.random.uniform(-1,1,[d,n]))
    self.b.append(np.random.uniform(-1,1,[n,1]))
    # self.dw.append(np.zeros((d,n)))
    # self.db.append(np.zeros((n,1)))

    # Hidden-Hidden Layer
    for i in range(1,L-2) : self.W.append(np.random.uniform(-1,1,[n,n]))
    for i in range(1,L-2) : self.b.append(np.random.uniform(-1,1,[n,1]))
    # for i in range(1,L-2) : self.dw.append(np.zeros((n,n)))
    # for i in range(1,L-2) : self.db.append(np.zeros((n,1)))

    # Hidden-Output Layer
    self.W.append(np.random.uniform(-1,1,[n,k]))
    self.b.append(np.random.uniform(-1,1,[k,1]))
    # self.dw.append(np.zeros((n,k)))
    # self.db.append(np.zeros((k,1)))

    self.flush_gradients()

    # Preactivation, Postactivation and y_hat vectors.
    # h size : (layer_size), total L+1 vectors
    # a size : (layer_size), total L+1 vectors
    # y_hat size : one single vecto, (N x k)

    self.a = [[]]
    self.h = []
    self.y_hat = np.zeros((k,1))

    # h[0] is input layer
    self.h.append(np.zeros((d,1)))

    # Appending vectors for L-layers
    for i in range(1,L-1): self.h.append(np.zeros((n,1)))
    for i in range(1,L-1): self.a.append(np.zeros((n,1)))

    # a[L] is output layer
    self.a.append(np.zeros((k,1)))

    self.LOSS = 0

    #   ______               _     ______                                     _ 
    #  |  ____|             | |   |  ____|                                   | |
    #  | |__  ___   ___   __| |   | |__  ___   _ __ __      __ __ _  _ __  __| |
    #  |  __|/ _ \ / _ \ / _` |   |  __|/ _ \ | '__|\ \ /\ / // _` || '__|/ _` |
    #  | |  |  __/|  __/| (_| |   | |  | (_) || |    \ V  V /| (_| || |  | (_| |
    #  |_|   \___| \___| \__,_|   |_|   \___/ |_|     \_/\_/  \__,_||_|   \__,_|

    # Feed Forwward function (model_instance, X[i], y[i]) : 
                                                                                                                                                                                        
  def forward(self, X):

    L = self.L

    # Setting Input Layer
    self.h[0] = (X/255).reshape(len(X),1)

    ## Forward pass for a data point X, and a label y

    for k in range(1,L-1):

      #print("k = {0}, a_({0}) = {2}, b_({0}) = {3}, W_({0}) = {4}, h_({1}) = {5}".format(k,k-1,self.a[k].shape,self.b[k].shape,self.W[k].T.shape,self.h[k-1].shape))
      self.a[k] = self.b[k] + (self.W[k].T @ self.h[k-1])
      #print(self.a[k])
      self.h[k] = functions[self.activation](self.a[k])
      #print(self.h[k])

    self.a[L-1] = self.b[L-1] + (self.W[L-1].T @ self.h[L-2])

    self.y_hat = functions[self.output](self.a[L-1])

    # self.LOSS += 

    # print(self.y_hat)

    return self.y_hat

    #   ____                _                                                    _    _               
    #  |  _ \              | |                                                  | |  (_)              
    #  | |_) |  __ _   ___ | | __ _ __   _ __  ___   _ __    __ _   __ _   __ _ | |_  _   ___   _ __  
    #  |  _ <  / _` | / __|| |/ /| '_ \ | '__|/ _ \ | '_ \  / _` | / _` | / _` || __|| | / _ \ | '_ \ 
    #  | |_) || (_| || (__ |   < | |_) || |  | (_) || |_) || (_| || (_| || (_| || |_ | || (_) || | | |
    #  |____/  \__,_| \___||_|\_\| .__/ |_|   \___/ | .__/  \__,_| \__, | \__,_| \__||_| \___/ |_| |_|
    #                            | |                | |             __/ |                             
    #                            |_|                |_|            |___/                              

  def backpropagate(self,y,y_hat):

    L = self.L
    k = self.k

    # Note : dw[0] and db[0] are both dummy variables
    dw_ = [i for i in range(L)]
    db_ = [i for i in range(L)]
    
    da = - (functions["e_l"](y,k)-y_hat)

    for i in range(L-1,0,-1):

      dw_[i] = self.h[i-1] @ da.T

      db_[i] = da

      if(i==1) : break

      dh = self.W[i] @ da

      da = dh * functions["d_"+self.activation](self.a[i-1])

    # print("dw_")
    # for i in range(1,L) : print(dw_[i].shape)
    # print("dw")
    # for i in range(1,L) : print(self.dw[i].shape)

    # print("db_")
    # for i in range(1,L) : print(db_[i].shape)
    # print("db")
    # for i in range(1,L) : print(self.db[i].shape)
      
    ###### Accumulate Gradients ######
    ##################################
    for i in range(1,L): self.dw[i] = np.add(self.dw[i],dw_[i])
    for i in range(1,L): self.db[i] = np.add(self.db[i],db_[i])
    ##################################
    ##################################

  def flush_gradients(self):

    d = self.d
    n = self.n
    k = self.k
    L = self.L

    # Set gradients back to zero
    self.dw = [[]]
    self.db = [[]]

    self.dw.append(np.zeros((d,n)))
    self.db.append(np.zeros((n,1)))

    for i in range(1,L-2) : self.dw.append(np.zeros((n,n)))
    for i in range(1,L-2) : self.db.append(np.zeros((n,1)))

    self.dw.append(np.zeros((n,k)))
    self.db.append(np.zeros((k,1)))

    #print("d: {}, n: {}, k: {}, L: {}".format(d,n,k,L))

    #   _____         _                                      
    #  |  __ \       | |                                     
    #  | |  | |  ___ | |__   _   _   __ _   __ _   ___  _ __ 
    #  | |  | | / _ \| '_ \ | | | | / _` | / _` | / _ \| '__|
    #  | |__| ||  __/| |_) || |_| || (_| || (_| ||  __/| |   
    #  |_____/  \___||_.__/  \__,_| \__, | \__, | \___||_|   
    #                                __/ |  __/ |            
    #                               |___/  |___/             

  def get_updates(self):

    return self.W,self.b,self.dw,self.db

  def set_weights(self,W):

    self.W = W
    

  def set_biases(self,b):

    self.b = b
    

  def describe(self):

    # print("---Architecture---\n")
    # print("Input layer ({0}).".format(self.d))
    # print("({0}) Hidden Layers ({1}).".format(self.L,self.n))
    # print("Activation Function ({0}).".format(self.activation))
    # print("Output layer ({0}).".format(self.k))
    # print("Output Function ({0}).".format(self.output))
    # print("Loss Function ({0}).\n".format(self.loss))

    print("---WandB---\n")
    print("Weights : ")
    for i in range(1,len(self.W)) :  print(self.W[i].shape)
    print("Biases : ")
    for i in range(1,len(self.b)) : print(self.b[i].shape)
    print("\n")

    # print("---Activation-Layers---\n")
    # print("pre-activation : ")
    # for i in range(1,len(self.a)) : print(self.a[i].shape)
    # print("Activation : ")
    # for i in self.h : print(i.shape)
    # print("\n")
