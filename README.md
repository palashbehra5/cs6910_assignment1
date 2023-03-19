# CS6910 - Assignment 1 - Neural Network Training Framework

- This project provides a Python-based framework for training neural networks on various datasets with customizable model, optimizer, and training parameters. The framework also provides support for tracking experiments using the Weights and Biases (wandb) platform.
Usage

- The framework can be run by executing the train.py script with the appropriate command-line arguments. The available arguments and their default values are as follows:
String type arguments

```
    -wp or --wandb_project: wandb project name (default: "myprojectname")
    -we or --wandb_entity: wandb entity name (default: "myname")
    -d or --dataset: name of the dataset (default: "mnist")
    -l or --loss: loss function to use (default: "mse")
    -o or --optimizer: optimizer to use (default: "momentum")
    -w_i or --weight_init: weight initialization method to use (default: "xavier")
    -a or --activation: activation function to use (default: "relu")
    -opt or --output: output activation function to use (default: "softmax")
```

- Integer type arguments

```
    -e or --epochs: number of epochs to train for (default: 10)
    -nhl or --num_layers: number of hidden layers in the model (default: 2)
    -sz or --hidden_size: number of neurons in each hidden layer (default: 128)
    -b or --batch_size: batch size to use for training (default: 4)
```

- Float type arguments

```
    -lr or --learning_rate: learning rate to use for optimization (default: 0.01)
    -m or --momentum: momentum parameter to use for optimization (default: 0.8)
    -beta: beta parameter to use for optimization (default: 0.9)
    -beta1: beta1 parameter to use for optimization (default: 0.99)
    -beta2: beta2 parameter to use for optimization (default: 0.99)
    -eps: epsilon parameter to use for optimization (default: 1e-8)
    -w_d or --weight_decay: weight decay parameter to use for optimization (default: 1e-5)
```

- Example usage

To run the framework with default parameters, simply execute the following command:

```
python train.py
```

- To customize the parameters, add the appropriate command-line arguments. For example, to train on the CIFAR-10 dataset using the Adam optimizer with a learning rate of 0.001 and a batch size of 16, execute the following command:

```
python train.py --dataset cifar10 --optimizer adam --learning_rate 0.001 --batch_size 16
```

Note : Currently only supports mnist and fashion_mnist, it is assumed new dataset has been imported and added to train.py

- Outputs

 The framework outputs the following during training:

    Average training loss for each epoch
    Average validation loss for each epoch (if validation data is provided)
    Training accuracy for each epoch
    Validation accuracy for each epoch (if validation data is provided)

- In addition, if wandb is enabled, the framework logs the above metrics to wandb, along with the model architecture and the final test accuracy.

# functions.py

This file contains various activation functions, loss functions, and metrics that are used in the neural network model.
Functions

    - sigmoid(z): computes the sigmoid function for the given input.
    - d_sigmoid(z): computes the derivative of the sigmoid function for the given input.
    - softmax(a_l): computes the softmax function for the given input.
    - relu(z): computes the rectified linear unit (ReLU) function for the given input.
    - d_relu(z): computes the derivative of the ReLU function for the given input.
    - tanh(z): computes the hyperbolic tangent function for the given input.
    - d_tanh(z): computes the derivative of the hyperbolic tangent function for the given input.
    - e_l(y, k): generates a one-hot encoded vector for the given class label and number of classes.
    - cross_entropy(y, y_hat): computes the cross-entropy loss between the predicted and actual values.
    - mse(y, y_hat): computes the mean squared error (MSE) between the predicted and actual values.
    - d_cross_entropy(y, k, y_hat): computes the derivative of the cross-entropy loss for backpropagation.
    - d_mse(y, k, y_hat): computes the derivative of the MSE loss for backpropagation.
    - softplus(z): computes the softplus function for the given input.
    - d_softplus(z): computes the derivative of the softplus function for the given input.
    - arctan(z): computes the arctangent function for the given input.
    - d_arctan(z): computes the derivative of the arctangent function for the given input.

These functions are stored in a dictionary called functions for easy access.

> Adding New Functions :

To add a new activation function, loss function, or metric, you can simply define the function and its derivative in this file, and then add it to the functions dictionary. Once added, the function can be easily used in the neural network model

# metrics.py

- The metrics.py module contains implementations of various evaluation metrics that can be used to assess the performance of a neural network model.

- The accuracy function calculates the accuracy of the predictions made by the model given the true labels. It returns a value between 0 and 1, with higher values indicating better performance.

- The confusion_matrix function calculates the confusion matrix, which is a table that summarizes the number of true positive, true negative, false positive, and false negative predictions made by the model. This function takes in the predicted labels and true labels, and returns a k x k matrix, where k is the number of classes.

- The e_l function is a helper function that returns a one-hot encoded vector for the given label y.

- The cross_entropy function calculates the cross-entropy loss between the predicted probability distribution y_hat and the true label y. This is a commonly used loss function for multi-class classification problems.

- The mse function calculates the mean squared error between the predicted probability distribution y_hat and the true label y. This is a commonly used loss function for regression problems.

Overall, the metrics.py module provides useful tools for evaluating the performance of a neural network model, both in terms of accuracy and loss. To add a new loss function, simply define it in the metric.py module, import it in functions and add it to the dictionary.

# model.py

- This is a simple neural network library implemented in Python, with support for both binary and multiclass classification problems. The library uses a feedforward architecture with a customizable number of hidden layers and activation functions. The user can choose between two weight initialization methods: random initialization or Xavier initialization.

- To use the library, first initialize a model object with the desired parameters, such as the number of hidden layers, hidden layer size, input layer size, output layer size, activation function, output function, loss function, and weight initialization method.

```
from model import model

params = {
    'num_layers': 2,
    'hidden_size': 16,
    'input_layer_size': 4,
    'output_layer_size': 2,
    'activation': 'sigmoid',
    'output': 'softmax',
    'loss': 'cross_entropy',
    'weight_init': 'random',
    'N': 1000
}

net = model(params)

```

# optimizer.py

- This is a code for an optimizer class. It contains several optimization algorithms like SGD, Momentum, RMSprop and Adam. The class is initialized with some hyperparameters that define the behavior of the optimizer, like learning rate, momentum, beta values, epsilon, weight decay, and optimizer type.

- The optimizer has two methods, update and optimize. The update method takes the model parameters, weights and biases, and their corresponding gradients as input, and updates the parameters according to the gradients. The optimize method takes the model, gradients, batch size, and optimizer type, and optimizes the model according to the specified optimizer.

- For each optimizer, the corresponding update rule is implemented in the optimize method. For example, for Momentum optimizer, the update rule is as follows:

```
W_update = [ gamma*W_update[i-1]+(eta * dw[i]) for i in range(1,len(W))]
b_update = [ gamma*b_update[i-1]+(eta * db[i]) for i in range(1,len(b))]
```

where gamma is the momentum value and eta is the learning rate.

- Nesterov accelerated optimizers can be found in Q4(b).py.

# train.py

To train the model, run the train.py script in the terminal or an IDE.

The following command can be used to run the script:

```
python train.py
```

# Future Improvements

   - Add support for other datasets
   - Implement different activation functions
   - Implement dropout regularization


