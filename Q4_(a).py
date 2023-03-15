import wandb
import numpy as np
from tensorflow.keras.datasets import fashion_mnist,mnist
from functions import functions
from model import model
from optimizer import optimizer
from metrics import accuracy

wandb.login()

sweep_config = {
    "method": "bayes",
    "metric":{
    "name": "val_accuracy",
    "goal": "maximize"
    },
    'parameters': {
        'num_epochs': {'values': [5,10]},
        'num_hidden_layers': {'values': [2, 3, 4]},
        'hidden_layer_size': {'values': [32, 64, 128]},
        'optimizer': {'values': ['adam','sgd','rmsprop','momentum']},
        'batch_size': {'values': [4, 16, 128]},
        'weight_initialisation': {'values': ['xavier','random']},
        'activation_function': {'values': ['sigmoid', 'tanh', 'relu','arctan','softplus']},
        'momentum' : {'values' : [0.8,0.9]},
        'loss' : {'values' : ['cross_entropy']},
        'weight_decay' : {'values' : [0,1e-3,1e-4,1e-5]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="vanilla_runs")

def train(config=None):
    # Initialize a new wandb run

    # with wandb.init() as run : 

    # Example sweep-run name :
    # hl_3_bs_16_ac_tanh_opt_sgd

    wandb.init(config=config)

    run_name = "hl_"+str(wandb.config.num_hidden_layers)+"_bs_"+str(wandb.config.batch_size)+"_ac_"+(wandb.config.activation_function)+"_opt_"+(wandb.config.optimizer)+"_hls_"+str(wandb.config.hidden_layer_size)+"_wd_"+str(wandb.config.weight_decay)

    wandb.run.name = run_name

    config = wandb.config

    if(wandb.config.optimizer=="sgd") : learning_rate = 1e-1
    elif(wandb.config.optimizer=="momentum") : learning_rate = 1e-2
    elif(wandb.config.optimizer=="rmsprop") : learning_rate = 1e-4
    elif(wandb.config.optimizer=="adam") : learning_rate = 1e-3

    learning_rate = learning_rate * (4**0.5)/(wandb.config.batch_size**0.5)

    model_params = {
    
        "loss": config.loss,
        "num_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_layer_size,
        "activation":	config.activation_function,
        "output": "softmax",
        "weight_init": config.weight_initialisation

    }

    training_params = {
            
        "epochs": config.num_epochs,
        "batch_size": config.batch_size,
        "dataset":"fashion_mnist"

    }

    optimizer_params = {
            
        "learning_rate": learning_rate,
        "momentum": config.momentum,
        "beta": 0.99,
        "beta1": 0.99,
        "beta2": 0.99,
        "epsilon": 	1e-8,
        "weight_decay": config.weight_decay,
        "optimizer": config.optimizer

    }

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
    x_test = x_test.reshape((x_test.shape[0],x_test.shape[1]*x_test.shape[2]))

    x_val = x_train[:6000]
    y_val = y_train[:6000]

    x_train = x_train[6000:]
    y_train = y_train[6000:]

    x_train = x_train/255
    x_val = x_val/255
    x_test = x_test/255

    # Number of data points
    N = len(x_train)
    N_val = len(x_val)

    # Shape of input layer
    d = x_train.shape[1]

    # Shape of output layer
    k = len(set(y_train))

    model_params['input_layer_size'] = d
    model_params['output_layer_size'] = k
    model_params['N'] = N

    nn = model(model_params)
    opt = optimizer(optimizer_params)

    epochs = training_params['epochs']
    batch_size = training_params['batch_size']

    dw,db = [],[]
    L = model_params['num_layers']+2

    ######### TRAINING MODEL ###########

    for e in range(1,epochs+1):

      train_loss = 0
      val_loss = 0
      curr = 0

      for i in range(len(x_train)):

        # Returns a,h,y_hat
        fw = nn.forward(x_train[i])

        train_loss += functions[model_params["loss"]](y_train[i],k,fw['y_hat'])
        if (i<N_val) : val_loss += functions[model_params["loss"]](y_val[i],k,fw['y_hat'])

        # Returns dw_,db_
        bw = nn.backpropagate(y_train[i],fw)
        curr+=1

        # Initializing Gradients
        if(curr == 1) : dw,db = bw['dw_'],bw['db_']

        # Accumulating Gradients
        else :

          for i in range(1,L) : dw[i]+=bw['dw_'][i]
          for i in range(1,L) : db[i]+=bw['db_'][i]

        if curr == batch_size:

          # Use gradients to optimize parameters
          opt.optimize(nn,dw,db,batch_size)
          curr = 0

      # Residue update
      if(curr>0): opt.optimize(nn,dw,db,batch_size)
          
      wandb.log({"train_loss": train_loss/N, "epoch": e, "val_loss" : val_loss/N_val}) 

    y_pred_train = [np.argmax(nn.forward(x_train[i])['y_hat']) for i in range(len(x_train))]
    y_pred_val = [np.argmax(nn.forward(x_val[i])['y_hat']) for i in range(len(x_val))]

    wandb.log({"train_accuracy": accuracy(y_pred_train,y_train), "val_accuracy": accuracy(y_pred_val,y_val)})         

wandb.agent(sweep_id, train, count = 50)