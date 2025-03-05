import numpy as np

def compute_cost(X, y, theta):
    # Initialize some useful values
    m = y.size

    # Compute the hypothesis (h(x) = X * theta)
    hyp = np.dot(X, theta)

    # Compute the cost function
    cost = np.sum(np.square(hyp - y)) / (2 * m)
    
   
    
    return cost

