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
                mu[i] = minimize(self.mu_function(i, mu, sigma, gamma), mu[i])
                sigma[i] = minimize(self.sigma_function(i, mu, sigma, gamma), sigma[i], bounds=[(1e-5, np.inf)])
                gamma_old[i] = gamma[i]
                gamma[i] = logit_inv(self.gamma_function(i, mu, sigma, gamma))
                
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


class LaplaceVB(BaseVB):

    def __init__(self, data):
        self.lambd = 1
        super().__init__(data)


    def mu_function(self, i, mu, sigma, gamma):
        mask = (np.arange(self.p) != i)
        def func(mu_i):
            terms = [
                mu_i * (self.XX[i, :] * gamma * mu)[mask].sum(),
                self.XX[i, i] * (mu_i**2) / 2,
                - self.YX[i] * mu_i,
                self.lambd * sigma[i] * np.sqrt(2 / np.pi) * np.exp(-mu_i**2 / (2 * sigma[i]**2)),
                self.lambd * mu_i * (1 - 2*sp.stats.norm.cdf(-mu_i / sigma[i]))
            ]
            return sum(terms)
        return func

    def sigma_function(self, i, mu, sigma, gamma):
        def func(sigma_i):
            terms = [
                self.XX[i, i] * (sigma_i**2) / 2,
                self.lambd * mu[i] * sigma_i * np.sqrt(2 / np.pi) * np.exp(-mu[i]**2 / (2 * sigma_i**2)),
                self.lambd * mu[i] * (1 - sp.stats.norm.cdf(mu[i] / sigma_i)),
                -np.log(sigma_i)
            ]
            return sum(terms)
        return func

    def gamma_function(self, i, mu, sigma, gamma):
        terms = [
            np.log(self.a0 / self.b0),
            np.log(np.sqrt(np.pi) * sigma[i] * self.lambd / np.sqrt(2)),
            -self.mu_function(i, mu, sigma, gamma)(mu[i]),
            -self.XX[i, i] * (sigma[i]**2) / 2,
            1/2
        ]
        return sum(terms)

