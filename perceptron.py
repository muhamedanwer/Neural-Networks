import numpy as np 


def perceptron(x):
    '''
    Args: x -> 
    output: return 1/1-e**-x
    '''
    return 1/(1- np.exp(-x))


def activation(x):
    return 