import numpy as np

def accuracy(y_pred,y_true):

    return sum(y_pred==y_true)/len(y_pred)

def confusion_matrix(y_pred,y_true):

    matrix = np.zeros((10,10))

    for i in range(len(y_true)):
        matrix[y_true[i]][y_pred[i]]+=1

    return matrix

def e_l(y,k):

    e_l = np.zeros((k,1))
    e_l[y][0] = 1
    return e_l

## Loss functions go here

def cross_entropy(y,k,y_hat):

    return -np.log(np.dot(e_l(y,k)),y_hat)

def mse(y,k,y_hat):

    return np.linalg.norm(e_l(y,k)-y_hat)**2