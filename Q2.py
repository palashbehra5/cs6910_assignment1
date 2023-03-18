from model import model
from tensorflow.keras.datasets import fashion_mnist
from interface import model_params
import numpy as np
import wandb

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape((x_train.shape[0],x_train.shape[1]*x_train.shape[2]))

wandb.init(project='fashion-mnist')
x_train = x_train/255

N = len(x_train)
d = x_train.shape[1]
k = len(set(y_train))

model_params['input_layer_size'] = d
model_params['output_layer_size'] = k
model_params['N'] = N

nn = model(model_params)

y_pred = np.array([nn.forward(x_train[i])['y_hat'] for i in range(len(x_train))])

probs = y_pred.mean(axis=0)
labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress','Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

data_table = wandb.Table(columns=["Class", "Probability"])
for i in range(len(labels)): data_table.add_data(labels[i], probs[i])
wandb.log({"predictions": wandb.plot.bar(data_table, "Class", "Probability", "Probability Distribution of Predictions")})