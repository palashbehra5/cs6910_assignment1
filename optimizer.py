from copy import deepcopy                                                                        

class optimizer:

  def __init__(self,params):

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
    self.t = 1

  # Update parameters with the "updates".
  def update(self, W, b, dw, db):

    # Check indexes carefully
    # Here dw,db are zero indexed.
    for i in range(1,len(W)): W[i] = W[i] - (dw[i-1])
    for i in range(1,len(b)): b[i] = b[i] - (db[i-1])


  ## Optimize model with gradients dw and db.
  ## dw,db are 1 indexed.
  ## updates (W.b) are 0 indexed.

  def optimize(self, model, dw, db, batch_size):

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
    t = self.t
    beta1 = self.beta1
    beta2 = self.beta2

    # Batch Size normalization
    dw = [dw[i]/batch_size for i in range(len(W))]
    db = [db[i]/batch_size for i in range(len(b))]

    # First Updates
    W_update = [ (eta * dw[i]) for i in range(1,len(W))]
    b_update = [ (eta * db[i]) for i in range(1,len(b))] 

    ## Add more optimizers here

    if(self.optimizer=="sgd"):

      self.init = 0

    elif(self.optimizer=="momentum") : 

      # First call to optimize
      if(self.init): self.init = 0

      else:

        W_update = [ gamma*W_update[i-1]+(eta * dw[i]) for i in range(1,len(W))]
        b_update = [ gamma*b_update[i-1]+(eta * db[i]) for i in range(1,len(b))]

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


    elif(self.optimizer=="adam"): 

        if(self.init):

          # 0-indexed, m_w,m_b,v_w,v_b
          m_w = [(1-beta1) * dw[i] for i in range(1,len(W))]
          m_b = [(1-beta1) * db[i] for i in range(1,len(b))]

          v_w = [(1-beta2) * (dw[i]**2) for i in range(1,len(W))]
          v_b = [(1-beta2) * (db[i]**2) for i in range(1,len(b))]

          self.init = 0
        
        else:

          m_w = [((beta1) * m_w[i-1]) + ((1-beta1) * dw[i]) for i in range(1,len(W))]
          m_b = [((beta1) * m_b[i-1]) + ((1-beta1) * db[i]) for i in range(1,len(b))]

          v_w = [((beta2) * v_w[i-1]) + ((1-beta2) * dw[i]**2) for i in range(1,len(W))]
          v_b = [((beta2) * v_b[i-1]) + ((1-beta2) * db[i]**2) for i in range(1,len(b))]

        m_hat_w = [ m_w[i-1] / (1-(beta1)**t) for i in range(1,len(W))]
        m_hat_b = [ m_b[i-1] / (1-(beta1)**t) for i in range(1,len(b))]

        v_hat_w = [ v_w[i-1] / (1-(beta2)**t) for i in range(1,len(W))]
        v_hat_b = [ v_b[i-1] / (1-(beta2)**t) for i in range(1,len(W))]

        W_update = [(eta * m_hat_w[i]) / (v_hat_w[i] + epsilon)**0.5 for i in range(len(m_hat_w))]
        b_update = [(eta * m_hat_b[i]) / (v_hat_b[i] + epsilon)**0.5 for i in range(len(m_hat_b))]

        self.t = t+1

    elif(self.optimizer=="more_optimizers_go_here"): return 1

    # Theta_(t+1) = Theta_t - (Theta_update)
    self.update(W,b,W_update,b_update)

    # Save Updates
    self.W_update = W_update
    self.b_update = b_update
    self.v_w = v_w
    self.v_b = v_b
    self.m_w = m_w
    self.m_b = m_b

    # Update model with new parameters
    model.W = W
    model.b = b