import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def forward_pass(x, weights, biases):
    """
    x: numpy array shape (1, n_features)
    weights: list of numpy arrays (in_dim, out_dim)
    biases: list of numpy arrays (1, out_dim)
    returns: numpy array output (1, out_dim) 
    """
    a = x
    for W, b in zip(weights, biases):
        a = sigmoid(np.dot(a, W) + b)   # shapes: (1,in)@(in,out) -> (1,out) + b (1,out)
    return a
