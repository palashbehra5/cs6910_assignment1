import numpy as np
import argparse
from tensorflow.keras.datasets import fashion_mnist,mnist
from model import model

parser = argparse.ArgumentParser(description='CS6910 - Assignment 1 :')

# String type arguments
parser.add_argument('-wp','--wandb_project', type=str, default = "myprojectname")
parser.add_argument('-we','--wandb_entity', type=str, default = "myname")
parser.add_argument('-d','--dataset', type=str, default = "fashion_mnist")
parser.add_argument('-l','--loss', type=str, default = "cross_entropy")
parser.add_argument('-o','--optimizer', type=str, default = "sgd")
parser.add_argument('-w_i','--weight_init', type=str, default = "random")
parser.add_argument('-a','--activation', type=str, default = "sigmoid")

# Integer type arguments
parser.add_argument('-e','--epochs', type=int, default = 1)
parser.add_argument('-nhl','--num_layers', type=int, default = 1)
parser.add_argument('-sz','--hidden_size', type=int, default = 4)
parser.add_argument('-b','--batch_size', type=int, default = 4)

# Float type arguments
parser.add_argument('-lr','--learning_rate',type=float, default = 0.1)
parser.add_argument('-m','--momentum',type=float, default = 0.5)
parser.add_argument('-beta','--beta',type=float, default = 0.5)
parser.add_argument('-beta1','--beta1',type=float, default = 0.5)
parser.add_argument('-beta2','--beta2',type=float, default = 0.5)
parser.add_argument('-eps','--epsilon',type=float, default = 0.000001)
parser.add_argument('-w_d','--weight_decay',type=float, default =.0)

args = parser.parse_args()

print("Initializing project : ",args.wandb_project,", By entity : ",args.wandb_entity,", for dataset : ",args.dataset)
print("epochs : ",args.epochs)
print("batch_size :",args.batch_size)
print("loss : ",args.loss)
print("optimizer : ",args.optimizer)
print("learning_rate : ",args.learning_rate)
print("momentum : ",args.momentum)
print("beta : ",args.beta)
print("beta1 : ",args.beta1)
print("beta2 : ",args.beta2)
print("epsilon : ",args.epsilon)
print("weight_decay : ",args.weight_decay)
print("weight_init : ",args.weight_init)
print("num_layers : ",args.num_layers)
print("hidden_size : ",args.hidden_size)
print("activation : ",args.activation)


# More datasets can be added in the following code

########################################### ADD BELOW ###########################################

if (args.dataset == "fashion_mnist") :  (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
elif (args.dataset == "mnist") :  (x_train, y_train), (x_test, y_test) = mnist.load_data()

#################################################################################################

params = {

    "dataset	":args.dataset,
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "loss": args.loss,
    "optimizer": args.optimizer,
    "learning_rate": args.learning_rate,
    "momentum": args.momentum,
    "beta": args.beta,
    "beta1": args.beta1,
    "beta2": args.beta2,
    "epsilon": 	args.epsilon,
    "weight_decay": args.weight_decay,
    "weight_init":args.weight_init,
    "num_layers": args.num_layers,
    "hidden_size": args.hidden_size,
    "activation": args.activation,
    "output": args.output

}

x_train = x_train.reshape((x_train.shape[0],x_train.shape[1]*x_train.shape[2]))
x_test = x_test.reshape((x_test.shape[0],x_test.shape[1]*x_test.shape[2]))

# Number of data points
N = len(x_train)

# Shape of input layer
d = x_train.shape[1]

# Shape of output layer
k = len(set(y_train))

params['input_layer_size'] = d
params['output_layer_size'] = k
params['N'] = N


# Creating an object of the neural network
nn1 = model(params,x_train,y_train)
nn1.describe()



