import numpy as np
from interface import parse
from tensorflow.keras.datasets import fashion_mnist,mnist
from model import model
from optimizer import optimizer
import seaborn as sns
from sklearn.metrics import confusion_matrix
from functions import functions,e_l
import matplotlib.pyplot as plt

if __name__ == "__main__":

  args = parse()

  # More datasets can be added in the following code

  ########################################### ADD BELOW ###########################################

  if (args.dataset == "fashion_mnist") :  (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
  elif (args.dataset == "mnist") :  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  #################################################################################################

  model_params = {

      
      "loss": args.loss,
      "weight_init":args.weight_init,
      "num_layers": args.num_layers,
      "hidden_size": args.hidden_size,
      "activation": args.activation,
      "output": args.output

  }

  optimizer_params = {

      "optimizer": args.optimizer,
      "learning_rate": args.learning_rate,
      "momentum": args.momentum,
      "beta": args.beta,
      "beta1": args.beta1,
      "beta2": args.beta2,
      "epsilon": 	args.epsilon,
      "weight_decay": args.weight_decay

  }

  training_params = {

      "epochs": args.epochs,
      "batch_size": args.batch_size,
      "dataset":args.dataset

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
      # print("A")
      # print(opt.W_update[1])
      # print("B")
      # print(dw[2]*1e-3)
      curr = 0

  # Residue update
  if(curr>0): opt.optimize(nn,dw,db,batch_size)

  print(nn.W[2])
  print(e,Loss/len(x_train))


plt.figure(figsize=(10,10))
y_pred = [np.argmax(nn.forward(x_train[i])['y_hat']) for i in range(len(x_train))]
sns.heatmap(confusion_matrix(y_train,y_pred),annot=True)
plt.show()