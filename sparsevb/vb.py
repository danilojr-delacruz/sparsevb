import numpy as np
import scipy as sp
import scipy.stats
import scipy.special
import time

from abc import ABC, abstractmethod
from sparsevb import *

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

    def __repr__(self):
        return "BaseVB()"

    def __str__(self):
        return self.__repr()


class GaussianVB(BaseVB):

    def mu_function(self, i, mu, sigma, gamma):
        mask = (np.arange(self.p) != i)
        return sigma[i]**2 * (self.YX[i] - (self.XX[i, :] * gamma * mu)[mask].sum())

    def sigma_function(self, i, mu, sigma, gamma):
        return 1 / np.sqrt(self.XX[i, i] + 1)

    def gamma_function(self, i, mu, sigma, gamma):
        return np.log(self.a0 / self.b0) + np.log(sigma[i]) + (mu[i]**2) / (2 * sigma[i]**2)


# Algorithm 3 - Batch, does not seem to converge well
#     def estimate_vb_parameters(self, tolerance=1e-5, verbose=False):
# 
#         mu, sigma, gamma = self.initial_values()
#         gamma_old = gamma.copy()
#         
#         delta_h = 1
#         # Does this need to be updated each time?
#         a = np.argsort(mu)[::-1]
#     
#         start_time = time.time()
#         epochs = 0
#         while delta_h >= tolerance:
#             print(epochs, round(delta_h, 5))
#             G = np.diag(gamma)
#             mu = np.linalg.inv(self.XX + G) @ (self.YX.T)
#             for i in range(self.p):
#                 gamma_old[i] = gamma[i]
#                 sigma[i] = self.sigma_function(i, mu, sigma, gamma)
#                 gamma[i] = logit_inv(self.gamma_function(i, mu, sigma, gamma))
# 
#             delta_h = DeltaH(gamma_old, gamma)
#             
#             epochs += 1
# 
#         end_time = time.time()
#         run_time = end_time - start_time
# 
#         if verbose:
#             print(f"Ran {epochs} epochs in {round(run_time, 4)} seconds.")
#             print(f"Final change in binary maximal entropy is {round(delta_h, 5)}.")
#             
#         return mu, sigma, gamma

# Algorithm 2
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
                gamma_old[i] = gamma[i]
                sigma[i] = self.sigma_function(i, mu, sigma, gamma)
                mu[i] = self.mu_function(i, mu, sigma, gamma)
                gamma[i] = logit_inv(self.gamma_function(i, mu, sigma, gamma))

            delta_h = DeltaH(gamma_old, gamma)
            
            epochs += 1

        end_time = time.time()
        run_time = end_time - start_time

        if verbose:
            print(f"Ran {epochs} epochs in {round(run_time, 4)} seconds.")
            print(f"Final change in binary maximal entropy is {round(delta_h, 5)}.")
            
        return mu, sigma, gamma

    def __repr__(self):
        return "Gaussian()"


class LaplaceVB(BaseVB):

    def __init__(self, data, lambd=1):
        self.lambd = lambd
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
                self.lambd * sigma_i * np.sqrt(2 / np.pi) * np.exp(-mu[i]**2 / (2 * sigma_i**2)),
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

    def __repr__(self):
        return f"Laplace(lambd={self.lambd})"


class FatLaplaceVB(BaseVB):

    def __init__(self, data, lambd=1, r=0.5):
        self.lambd = lambd
        self.r = r
        super().__init__(data)


    def rth_moment(self, mu, sigma):
        r = self.r

        term0 = (sigma**r) * (2**(r/2))
        term1 = sp.special.gamma((r+1) / 2) / sp.special.gamma(1/2)
        term2 = sp.special.hyp1f1(-r/2, 1/2, -1/2 * (mu / sigma)**2)
        
        return term0 * term1 * term2

    def mu_function(self, i, mu, sigma, gamma):
        mask = (np.arange(self.p) != i)
        def func(mu_i):
            terms = [
                mu_i * (self.XX[i, :] * gamma * mu)[mask].sum(),
                self.XX[i, i] * (mu_i**2) / 2,
                - self.YX[i] * mu_i,
                self.lambd * self.rth_moment(mu_i, sigma[i]) 
            ]
            return sum(terms)
        return func

    def sigma_function(self, i, mu, sigma, gamma):
        def func(sigma_i):
            terms = [
                self.XX[i, i] * (sigma_i**2) / 2,
                self.lambd * self.rth_moment(mu[i], sigma_i)
                -np.log(sigma_i)
            ]
            return sum(terms)
        return func

    def gamma_function(self, i, mu, sigma, gamma):
        r = self.r
        prior_normalising_factor = r * self.lambd**(1/r) / (2 * sp.special.gamma(1/r))
        terms = [
            np.log(self.a0 / self.b0),
            np.log(np.sqrt(2 * np.pi) * sigma[i] * prior_normalising_factor),
            -self.mu_function(i, mu, sigma, gamma)(mu[i]),
            -self.XX[i, i] * (sigma[i]**2) / 2,
            1/2
        ]
        return sum(terms)

    def __repr__(self):
        return f"FatLaplace(lambd={self.lambd}, r={self.r})"


class JensenBoundCauchy(BaseVB):

    def mu_function(self, i, mu, sigma, gamma):
        mask = (np.arange(self.p) != i)
        def func(mu_i):
            terms = [
                mu_i * (self.XX[i, :] * gamma * mu)[mask].sum(),
                self.XX[i, i] * (mu_i**2) / 2,
                - self.YX[i] * mu_i,
                np.log(np.pi) + np.log(1 + mu_i**2 + sigma[i]**2)
            ]
            return sum(terms)
        return func

    def sigma_function(self, i, mu, sigma, gamma):
        def func(sigma_i):
            terms = [
                self.XX[i, i] * (sigma_i**2) / 2,
                np.log(1 + mu[i]**2 + sigma_i**2),
                -np.log(sigma_i)
            ]
            return sum(terms)
        return func

    def gamma_function(self, i, mu, sigma, gamma):
        terms = [
            np.log(self.a0 / self.b0),
            -self.mu_function(i, mu, sigma, gamma)(mu[i]),
            -self.XX[i, i] * (sigma[i]**2) / 2,
            (1 + np.log(2*np.pi) )/ 2,
            -np.log(sigma[i])
        ]
        return sum(terms)


    def __repr__(self):
        return "JensenBoundCauchy()"


class NumericIntCauchy(BaseVB):

    @staticmethod
    @np.vectorize
    def neg_expected_log_g(mu, sigma):
        def A(t):
            term1 = np.log(1 + (t+mu)**2)
            term2 = np.exp(-(t/sigma)**2 / 2) / (sigma * np.sqrt(2*np.pi))
            return term1*term2
        
        def B(t):
            return A(1/t) / t**2

        left = sp.integrate.quad(B, -1, 0)[0]
        middle = sp.integrate.quad(A, -1, 1)[0]
        right = sp.integrate.quad(B, 0, 1)[0]

        return left + middle + right + np.log(np.pi)

    def mu_function(self, i, mu, sigma, gamma):
        mask = (np.arange(self.p) != i)
        def func(mu_i):
            terms = [
                mu_i * (self.XX[i, :] * gamma * mu)[mask].sum(),
                self.XX[i, i] * (mu_i**2) / 2,
                - self.YX[i] * mu_i,
                self.neg_expected_log_g(mu_i, sigma[i])
            ]
            return sum(terms)
        return func

    def sigma_function(self, i, mu, sigma, gamma):
        def func(sigma_i):
            terms = [
                self.XX[i, i] * (sigma_i**2) / 2,
                -np.log(sigma_i),
                self.neg_expected_log_g(mu[i], sigma_i)
            ]
            return sum(terms)
        return func

    def gamma_function(self, i, mu, sigma, gamma):
        terms = [
            np.log(self.a0 / self.b0),
            -self.mu_function(i, mu, sigma, gamma)(mu[i]),
            -self.XX[i, i] * (sigma[i]**2) / 2,
            (1 + np.log(2*np.pi) )/ 2,
            -np.log(sigma[i])
        ]
        return sum(terms)


    def __repr__(self):
        return "NumericIntCauchy()"

