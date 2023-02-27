import numpy as np

def accuracy(y_pred,y_true):

    return sum(y_pred==y_true)/len(y_pred)

def confusion_matrix(y_pred,y_true):

    matrix = np.zeros((10,10))
    for i in range(len(y_true)): matrix[y_true[i]][y_pred[i]]+=1

    return matrix
