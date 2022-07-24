import numpy as np
import scipy as sp


# Data Generation
def generate_data(X, theta):
    Z = np.random.normal(size=X.shape[0])
    Y = X@theta + Z
    
    return (X, Y)


# Entropy Functions
def H(x):
    """Return entropy of vector"""
    # Use np.log(x**x) so x=0 returns 0 and not nan
    return -np.log(x**x) -np.log((1-x)**(1-x))


def DeltaH(a, b):
    """Get the max (amongst components) difference in entropy"""
    return max(abs(H(a) - H(b)))


# Logit functions
def logit(x):
    return np.log(x / (1-x))


def logit_inv(x):
    return 1 / (1 + np.exp(-x))


# Optimisation
def minimize(fun, x0, *args, **kwargs):
    result = sp.optimize.minimize(fun, x0, *args, **kwargs)
    return result.x[0]


# Metrics
def l2(x, y):
    return np.sqrt(((x-y)**2).sum())


def fdr_tpr(underlying, observed):
    
    tp = (underlying & observed).sum()
    tn = ((~underlying) & (~observed)).sum()
    fp = ((~underlying) & observed).sum()
    fn = (underlying & (~observed)).sum()
    
    fdr = fp / (fp + tp)
    tpr = tp / (tp + fn)
    
    return fdr, tpr
