import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from model import model
from optimizer import optimizer
from tqdm import tqdm
from metrics import accuracy,confusion_matrix
from functions import functions
import wandb
import plotly.graph_objs as go



wandb.login()
wandb.init(project='Q7-A1')

model_params = {

    "loss": "cross_entropy",
    "weight_init":"xavier",
    "num_layers": 2,
    "hidden_size": 128,
    "activation": "relu",
    "output": "softmax"

  }

optimizer_params = {

    "optimizer": "momentum",
    "learning_rate": 1e-2,
    "momentum": 0.8,
    "beta": 0.9,
    "beta1": 0.9,
    "beta2": 0.9,
    "epsilon": 	1e-8,
    "weight_decay": 1e-5

  }

training_params = {

    "epochs": 10,
    "batch_size": 4,
    "dataset":"fashion_mnist"

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

y_pred_train = np.array([np.argmax(nn.forward(x_train[i])['y_hat']) for i in range(len(x_train))])
y_pred_val = np.array([np.argmax(nn.forward(x_val[i])['y_hat']) for i in range(len(x_val))])
y_pred_test = np.array([np.argmax(nn.forward(x_test[i])['y_hat']) for i in range(len(x_test))])

wandb.log({"train_accuracy": accuracy(y_pred_train,y_train), "val_accuracy": accuracy(y_pred_val,y_val), "test_accuracy": accuracy(y_pred_test,y_test)})   

conf_matrix = confusion_matrix(y_pred_test,y_test)

k = conf_matrix.shape[0]
trace = go.Heatmap(
    z=conf_matrix,
    x=np.arange(k)+1,
    y=np.arange(k)+1,
    colorscale='Blues',
    textfont=dict(color='black') 
)

annotations = []
for i in range(k):
    for j in range(k):
        annotations.append(
            dict(
                x=i+1,
                y=j+1,
                text=str(conf_matrix[j][i]), 
                showarrow=False
            )
        )

layout = go.Layout(
    title='Confusion Matrix',
    xaxis=dict(
        title='Predicted label',
        tickmode='array',
        tickvals=np.arange(k)+1,
        ticktext=np.arange(k)+1
    ),
    yaxis=dict(
        title='True label',
        tickmode='array',
        tickvals=np.arange(k)+1,
        ticktext=np.arange(k)+1
    ),
    annotations=annotations
)

fig = go.Figure(data=[trace], layout=layout)
wandb.log({"Confusion Matrix": fig})