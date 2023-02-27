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
    self.W_update = []
    self.b_update = []
    self.init = 1
    self.v_w = []
    self.m_w = []
    self.m_b = []
    self.v_b = []

  # Update parameters with the "updates".
  def update(self,W,b,dw,db,batch_size):

    # Check indexes carefully
    # Here dw,db are zero indexed.
    for i in range(1,len(W)): W[i] = W[i] - (dw[i-1]/batch_size)
    for i in range(1,len(b)): b[i] = b[i] - (db[i-1]/batch_size)


  ## Optimize model with gradients dw and db.
  ## dw,db are 1 indexed.
  ## updates (W.b) are 0 indexed.

  def optimize(self,model,dw,db,batch_size):

    W_update = self.W_update
    b_update = self.b_update
    v_w = self.v_w
    v_b = self.v_b
    m_w = self.m_w
    m_b = self.m_b
    eta = self.learning_rate
    W,b= model.W,model.b
    gamma = self.momentum
    beta = self.beta
    epsilon = self.epsilon

    ## Add more optimizers here

    if(self.optimizer=="sgd"):

      W_update = [ (eta * dw[i]) for i in range(1,len(W))]
      b_update = [ (eta * db[i]) for i in range(1,len(b))]

    elif(self.optimizer=="momentum") : 

      # First call to optimize
      if(self.init):

        W_update = [ (eta * dw[i]) for i in range(1,len(W))]
        b_update = [ (eta * db[i]) for i in range(1,len(b))]
        self.init = 0

      else:

        W_update = [ gamma*W_update[i-1]+(eta * dw[i]) for i in range(1,len(W))]
        b_update = [ gamma*b_update[i-1]+(eta * db[i]) for i in range(1,len(b))]

    elif(self.optimizer=="nag"): 

      # First call to optimize
      if(self.init):

        W_look_ahead = dw
        b_look_ahead = db
        W_update = [ (eta * W_look_ahead[i]) for i in range(1,len(W))]
        b_update = [ (eta * b_look_ahead[i]) for i in range(1,len(b))]
        self.init = 0

      else:

        W_look_ahead = [ (dw[i]-gamma*W_update[i-1]) for i in range(1,len(W))]
        b_look_ahead = [ (db[i]-gamma*b_update[i-1]) for i in range(1,len(b))]
        W_update = [ gamma*W_update[i-1]+(eta * W_look_ahead[i-1]) for i in range(1,len(W))]
        b_update = [ gamma*b_update[i-1]+(eta * b_look_ahead[i-1]) for i in range(1,len(b))]
      

    elif(self.optimizer=="rmsprop"): 

      # First call to optimize
      if(self.init):

        v_w = [(1-beta)*(dw[i]**2) for i in range(1,len(dw))]
        v_b = [(1-beta)*(db[i]**2) for i in range(1,len(db))]
        W_update = [ (eta*dw[i]) / (v_w[i-1]+epsilon)**0.5 for i in range(1,len(dw)) ]
        b_update = [ (eta*db[i]) / (v_b[i-1]+epsilon)**0.5 for i in range(1,len(db)) ]
        self.init = 0

      else:

        v_w = [ (beta*v_w[i-1]) + (1-beta)*(dw[i]**2) for i in range(1,len(dw))]
        v_b = [ (beta*v_b[i-1]) + (1-beta)*(db[i]**2) for i in range(1,len(db))]
        W_update = [ (eta*dw[i]) / (v_w[i-1]+epsilon)**0.5 for i in range(1,len(dw)) ]
        b_update = [ (eta*db[i]) / (v_b[i-1]+epsilon)**0.5 for i in range(1,len(db)) ]


    elif(self.optimizer=="adam"): return 1
  
    elif(self.optimizer=="nadam"): return 1

    elif(self.optimizer=="more_optimizers_go_here"): return 1

    # Theta_(t+1) = Theta_t - (update/batch_size)
    self.update(W,b,W_update,b_update,batch_size)

    # Save Updates
    self.W_update = W_update
    self.b_update = b_update
    self.v_w = v_w
    self.v_b = v_b

    # Update model with new parameters
    model.W = W
    model.b = b