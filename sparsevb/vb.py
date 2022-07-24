import numpy as np
import scipy as sp
import scipy.stats
import scipy.special
import time

from abc import ABC, abstractmethod


class BaseVB(ABC):

    def __init__(self, data):
        self.data = data
        X, Y = self.data
        self.XX = X.T @ X
        self.YX = Y.T @ X
        self.n, self.p = X.shape

        self.a0, self.b0 = 1, self.p
    
    def initial_values(self):

        X, Y = self.data
        # Ridge Regression Estimate
        mu = np.linalg.inv((X.T@X + np.eye(self.p))) @ X.T @ Y
        # How do we initialise this?
        sigma = np.ones(self.p)
        # How do we initialise this?
        gamma = np.ones(self.p) / 3
        
        return mu, sigma, gamma 

    @abstractmethod
    def mu_function(self, i, mu, sigma, gamma):
        pass

    @abstractmethod
    def sigma_function(self, i, mu, sigma, gamma):
        pass

    @abstractmethod
    def gamma_function(self, i, mu, sigma, gamma):
        pass

    def estimate_vb_parameters(self, tolerance=1e-5, verbose=False):

        mu, sigma, gamma = self.initial_values()
        gamma_old = gamma.copy()
        
        delta_h = 1
        # Does this need to be updated each time?
        a = np.argsort(mu)[::-1]
        
        start_time = time.time()
        epochs = 0
        while delta_h >= tolerance:
            for i in a:
                # Use old values throughout or newest as possible?
                mu[i] = minimize(self.f(i, mu, sigma, gamma), mu[i])
                sigma[i] = minimize(self.g(i, mu, sigma, gamma), sigma[i], bounds=[(1e-5, np.inf)])
                gamma_old[i] = gamma[i]
                gamma[i] = logit_inv(self.gamma(i, mu, sigma, gamma))
                
            delta_h = DeltaH(gamma_old, gamma)
            
            epochs += 1

        end_time = time.time()
        run_time = end_time - start_time

        if verbose:
            print(f"Ran {epochs} epochs in {round(run_time, 4)} seconds.")
            print(f"Final change in binary maximal entropy is {round(delta_h, 5)}.")
            
        return mu, sigma, gamma

    @staticmethod
    def posterior_mean(mu, gamma):
        return mu*gamma

