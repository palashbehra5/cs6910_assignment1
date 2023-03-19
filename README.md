# cs6910_assignment1
CS6910 Deep Learning Assignment 1

Functions

    sigmoid(z): computes the sigmoid function for the given input.
    d_sigmoid(z): computes the derivative of the sigmoid function for the given input.
    softmax(a_l): computes the softmax function for the given input.
    relu(z): computes the rectified linear unit (ReLU) function for the given input.
    d_relu(z): computes the derivative of the ReLU function for the given input.
    tanh(z): computes the hyperbolic tangent function for the given input.
    d_tanh(z): computes the derivative of the hyperbolic tangent function for the given input.
    e_l(y, k): generates a one-hot encoded vector for the given class label and number of classes.
    cross_entropy(y, y_hat): computes the cross-entropy loss between the predicted and actual values.
    mse(y, y_hat): computes the mean squared error (MSE) between the predicted and actual values.
    d_cross_entropy(y, k, y_hat): computes the derivative of the cross-entropy loss for backpropagation.
    d_mse(y, k, y_hat): computes the derivative of the MSE loss for backpropagation.
    softplus(z): computes the softplus function for the given input.
    d_softplus(z): computes the derivative of the softplus function for the given input.
    arctan(z): computes the arctangent function for the given input.
    d_arctan(z): computes the derivative of the arctangent function for the given input.

These functions are stored in a dictionary called functions for easy access.
Adding New Functions

To add a new activation function, loss function, or metric, you can simply define the function and its derivative in this file, and then add it to the functions dictionary. Once added, the function can be easily used in the neural network model.


