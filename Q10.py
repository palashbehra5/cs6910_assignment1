import wandb
import numpy as np
from tensorflow.keras.datasets import mnist
from functions import functions
from model import model
from optimizer import optimizer
from metrics import accuracy

wandb.login()


sweep_config_1 = {
    'method': 'grid',
    'parameters': {
        'activation_function': {'values': ['relu']},
        'batch_size': {'values': [4]},
        'hidden_layer_size': {'values': [128]},
        'loss': {'values': ['cross_entropy']},
        'momentum': {'values': [0.8]},
        'num_epochs': {'values': [10]},
        'num_hidden_layers': {'values': [2]},
        'optimizer': {'values': ['momentum']},
        'weight_decay': {'values': [1e-5]},
        'weight_initialisation': {'values': ['xavier']},
        'learning_rate' : {'values' : [1e-2]}
    }
}

sweep_config_2 = {
    'method': 'grid',
    'parameters': {
        'activation_function': {'values': ['relu']},
        'batch_size': {'values': [4]},
        'hidden_layer_size': {'values': [64]},
        'loss': {'values': ['cross_entropy']},
        'momentum': {'values': [0.9]},
        'num_epochs': {'values': [10]},
        'num_hidden_layers': {'values': [3]},
        'optimizer': {'values': ['rmsprop']},
        'weight_decay': {'values': [1e-5]},
        'weight_initialisation': {'values': ['xavier']},
        'learning_rate' : {'values' : [1e-4]}
    }
}

sweep_config_3 = {
    'method': 'grid',
    'parameters': {
        'activation_function': {'values': ['tanh']},
        'batch_size': {'values': [16]},
        'hidden_layer_size': {'values': [64]},
        'loss': {'values': ['cross_entropy']},
        'momentum': {'values': [0.8]},
        'num_epochs': {'values': [10]},
        'num_hidden_layers': {'values': [3]},
        'optimizer': {'values': ['adam']},
        'weight_decay': {'values': [1e-3]},
        'weight_initialisation': {'values': ['xavier']},
        'learning_rate' : {'values' : [0.5e-3]}
    }
}

sweep_id_1 = wandb.sweep(sweep_config_1, project="Q10")
sweep_id_2 = wandb.sweep(sweep_config_2, project="Q10")
sweep_id_3 = wandb.sweep(sweep_config_3, project="Q10")

def train(config=None):
    # Initialize a new wandb run

    # with wandb.init() as run : 

    # Example sweep-run name :
    # hl_3_bs_16_ac_tanh_opt_sgd

    wandb.init(config=config)

    run_name = "hl_"+str(wandb.config.num_hidden_layers)+"_bs_"+str(wandb.config.batch_size)+"_ac_"+(wandb.config.activation_function)+"_opt_"+(wandb.config.optimizer)+"_hls_"+str(wandb.config.hidden_layer_size)+"_wd_"+str(wandb.config.weight_decay)

    wandb.run.name = run_name

    config = wandb.config

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
            
        "learning_rate": config.learning_rate,
        "momentum": config.momentum,
        "beta": 0.99,
        "beta1": 0.99,
        "beta2": 0.99,
        "epsilon": 	1e-8,
        "weight_decay": config.weight_decay,
        "optimizer": config.optimizer

    }

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
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
      curr = 0

      for i in range(len(x_train)):

        # Returns a,h,y_hat
        fw = nn.forward(x_train[i])

        train_loss += functions[model_params["loss"]](y_train[i],k,fw['y_hat'])

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
          
      wandb.log({"train_loss": train_loss/N, "epoch": e}) 

      y_pred_train = np.array([np.argmax(nn.forward(x_train[i])['y_hat']) for i in range(len(x_train))])
      y_pred_val = np.array([np.argmax(nn.forward(x_val[i])['y_hat']) for i in range(len(x_val))])

      wandb.log({"train_accuracy": accuracy(y_pred_train,y_train), "val_accuracy": accuracy(y_pred_val,y_val)})          

    y_pred_test = np.array([np.argmax(nn.forward(x_test[i])['y_hat']) for i in range(len(x_test))])
    wandb.log({"test_accuracy": accuracy(y_pred_test,y_test)}) 

wandb.agent(sweep_id_1, function=train)
wandb.agent(sweep_id_2, function=train)
wandb.agent(sweep_id_3, function=train)