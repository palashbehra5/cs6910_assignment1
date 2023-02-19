# optimizer.py                                                                            

class optimizer:

  def __init__(self,params):

    #    ____          _    _             _                     _____                                     _                   
    #   / __ \        | |  (_)           (_)                   |  __ \                                   | |                  
    #  | |  | | _ __  | |_  _  _ __ ___   _  ____ ___  _ __    | |__) |__ _  _ __  __ _  _ __ ___    ___ | |_  ___  _ __  ___ 
    #  | |  | || '_ \ | __|| || '_ ` _ \ | ||_  // _ \| '__|   |  ___// _` || '__|/ _` || '_ ` _ \  / _ \| __|/ _ \| '__|/ __|
    #  | |__| || |_) || |_ | || | | | | || | / /|  __/| |      | |   | (_| || |  | (_| || | | | | ||  __/| |_|  __/| |   \__ \
    #   \____/ | .__/  \__||_||_| |_| |_||_|/___|\___||_|      |_|    \__,_||_|   \__,_||_| |_| |_| \___| \__|\___||_|   |___/
    #          | |                                                                                                            
    #          |_|                                                                                                            

    self.learning_rate = params['learning_rate']

    self.momentum = params['momentum']

    self.beta = params['beta']

    self.beta1 = params['beta1']

    self.beta2 = params['beta2']

    self.epsilon = params['epsilon']

    self.weight_decay = params['weight_decay']

    self.optimizer = params['optimizer']

  ## Optimize model with gradients dw and db.

  def optimize(self,model):

    ## Add more optimizers here

    if(self.optimizer=="sgd"):

      eta = self.learning_rate
      W,b,dw,db = model.get_updates()

      # for i in range(1,len(W)) : print(W[i].shape)
      # print("----")
      # for i in range(1,len(b)) : print(b[i].shape)
       
      for i in range(1,len(W)): W[i] = W[i] - (eta * dw[i])
      for i in range(1,len(b)): b[i] = b[i] - (eta * db[i])

      model.set_weights(W)
      model.set_biases(b)

      #print("OPTIMIZED")

    elif(self.optimizer=="momentum") : return 1

    elif(self.optimizer=="nag"): return 1

    elif(self.optimizer=="rmsprop"): return 1

    elif(self.optimizer=="adam"): return 1
  
    elif(self.optimizer=="nadam"): return 1

    elif(self.optimizer=="more_optimizers_go_here"): return 1