# CS6910 - Assignment 1 - Neural Network Training Framework

> This project provides a Python-based framework for training neural networks on various datasets with customizable model, optimizer, and training parameters. The framework also provides support for tracking experiments using the Weights and Biases (wandb) platform.
Usage

> The framework can be run by executing the train.py script with the appropriate command-line arguments. The available arguments and their default values are as follows:
String type arguments

    -wp or --wandb_project: wandb project name (default: "myprojectname")
    -we or --wandb_entity: wandb entity name (default: "myname")
    -d or --dataset: name of the dataset (default: "mnist")
    -l or --loss: loss function to use (default: "mse")
    -o or --optimizer: optimizer to use (default: "momentum")
    -w_i or --weight_init: weight initialization method to use (default: "xavier")
    -a or --activation: activation function to use (default: "relu")
    -opt or --output: output activation function to use (default: "softmax")

> Integer type arguments

    -e or --epochs: number of epochs to train for (default: 100)
    -nhl or --num_layers: number of hidden layers in the model (default: 2)
    -sz or --hidden_size: number of neurons in each hidden layer (default: 128)
    -b or --batch_size: batch size to use for training (default: 4)

> Float type arguments

    -lr or --learning_rate: learning rate to use for optimization (default: 0.01)
    -m or --momentum: momentum parameter to use for optimization (default: 0.8)
    -beta: beta parameter to use for optimization (default: 0.9)
    -beta1: beta1 parameter to use for optimization (default: 0.99)
    -beta2: beta2 parameter to use for optimization (default: 0.99)
    -eps: epsilon parameter to use for optimization (default: 1e-8)
    -w_d or --weight_decay: weight decay parameter to use for optimization (default: 1e-5)

> Example usage

To run the framework with default parameters, simply execute the following command:

python train.py

> To customize the parameters, add the appropriate command-line arguments. For example, to train on the CIFAR-10 dataset using the Adam optimizer with a learning rate of 0.001 and a batch size of 16, execute the following command:

python train.py --dataset cifar10 --optimizer adam --learning_rate 0.001 --batch_size 16

> Outputs

> The framework outputs the following during training:

    Average training loss for each epoch
    Average validation loss for each epoch (if validation data is provided)
    Training accuracy for each epoch
    Validation accuracy for each epoch (if validation data is provided)

> In addition, if wandb is enabled, the framework logs the above metrics to wandb, along with the model architecture and the final test accuracy.

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

Adding New Functions :

To add a new activation function, loss function, or metric, you can simply define the function and its derivative in this file, and then add it to the functions dictionary. Once added, the function can be easily used in the neural network model


