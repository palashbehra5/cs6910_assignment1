import numpy as np
from interface import model_params,optimizer_params,training_params
from tensorflow.keras.datasets import fashion_mnist,mnist
from model import model
from optimizer import optimizer
from tqdm import tqdm
from metrics import accuracy
from interface import model_params
from functions import functions


# More datasets can be added in the following code

########################################### ADD BELOW ###########################################

if (training_params['dataset'] == "fashion_mnist") :  (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
elif (training_params['dataset'] == "mnist") :  (x_train, y_train), (x_test, y_test) = mnist.load_data()

#################################################################################################

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

  for i in tqdm(range(len(x_train))):

    # Returns a,h,y_hat
    fw = nn.forward(x_train[i])

    Loss += functions[model_params["loss"]](y_train[i],k,fw['y_hat'])

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
      # Using forward results for nag and nadam
      opt.optimize(nn,dw,db,batch_size)
      curr = 0

  # Residue update
  if(curr>0): opt.optimize(nn,dw,db,batch_size)
  print(e,Loss/len(x_train))

y_pred = [np.argmax(nn.forward(x_train[i])['y_hat']) for i in range(len(x_train))]
print("Training Accuracy ",accuracy(y_pred,y_train))  

y_pred = [np.argmax(nn.forward(x_val[i])['y_hat']) for i in range(len(x_val))]
print("Validation Accuracy ",accuracy(y_pred,y_val))  

y_pred = [np.argmax(nn.forward(x_test[i])['y_hat']) for i in range(len(x_test))]
print("Testing Accuracy ",accuracy(y_pred,y_test)) 