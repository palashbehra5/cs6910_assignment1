import wandb
from wandb.keras import WandbCallback
from tensorflow.keras.datasets import fashion_mnist


wandb.init(project='fashion-mnist')

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()


labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

table = wandb.Table(columns=["Image", "Class"])


for i, labels in enumerate(labels):
    idx = next(j for j, label in enumerate(y_train) if label == i)
    table.add_data(wandb.Image(X_train[idx]), labels)

wandb.log({"examples": table})
callback = WandbCallback()
