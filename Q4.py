import wandb
import numpy as np
from tensorflow.keras.datasets import fashion_mnist,mnist
from model import model
from optimizer import optimizer
from functions import functions,e_l
from metrics import accuracy,confusion_matrix

wandb.login()

sweep_config = {
    'method': 'random',
    'parameters': {
        'num_epochs': {'values': [10, 15, 20]},
        'num_hidden_layers': {'values': [2, 3]},
        'hidden_layer_size': {'values': [32, 64, 128]},
        'learning_rate': {'values': [1e-1, 1e-2]},
        'optimizer': {'values': ['sgd', 'momentum']},
        'batch_size': {'values': [4, 32, 256]},
        'weight_initialisation': {'values': ['xavier']},
        'activation_function': {'values': ['sigmoid', 'tanh', 'relu']},
        'momentum' : {'values' : [0.7,0.8]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="assignment1_draft")

def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        
        config = wandb.config

        model_params = {
    
            "loss": "cross_entropy",
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
            "beta": 0.5,
            "beta1": 0.5,
            "beta2": 0.5,
            "epsilon": 	0.00001,
            "weight_decay": 0,
            "optimizer": config.optimizer

        }

        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = x_train.reshape((x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
        x_test = x_test.reshape((x_test.shape[0],x_test.shape[1]*x_test.shape[2]))

        x_train = x_train/255

        # Number of data points
        N = len(x_train)

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

          Loss = 0
          curr = 0
          updates = 0

          for i in range(len(x_train)):

            # Returns a,h,y_hat
            fw = nn.forward(x_train[i])

            Loss += -np.log((e_l(y_train[i],k).T @ fw['y_hat'])[0][0]) 

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
          
          wandb.log({"loss": Loss/N, "epoch": e}) 

        y_pred_train = [np.argmax(nn.forward(x_train[i])['y_hat']) for i in range(len(x_train))]
        y_pred_test = [np.argmax(nn.forward(x_test[i])['y_hat']) for i in range(len(x_test))]

        wandb.log({"train_accuracy": accuracy(y_pred_train,y_train), "test_accuracy": accuracy(y_pred_test,y_test)})         

wandb.agent(sweep_id, train, count = 20)