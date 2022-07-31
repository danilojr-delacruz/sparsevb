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

    def likelihood(self, i, mu_i, sigma_i, mu, sigma, gamma):
        mask = (np.arange(self.p) != i)
        value = mu_i * (self.XX[i, :] * gamma * mu * mask).sum() \
                + self.XX[i, i] * (sigma_i**2 + mu_i**2) / 2 \
                - self.YX[i] * mu_i
        return value

    def expected_log_prior(self, i, mu_i, sigma_i, mu, sigma, gamma):
        return 0

    def mu_function(self, i, mu, sigma, gamma):
        def func(mu_i):
            value = self.likelihood(i, mu_i, sigma[i], mu, sigma, gamma) \
                    - self.expected_log_prior(i, mu_i, sigma[i], mu, sigma, gamma)
            return value
        return func

    def sigma_function(self, i, mu, sigma, gamma):
        def func(sigma_i):
            value = self.likelihood(i, mu[i], sigma_i, mu, sigma, gamma) \
                    - np.log(sigma_i) \
                    - self.expected_log_prior(i, mu[i], sigma_i, mu, sigma, gamma)
            return value
        return func

    def gamma_function(self, i, mu, sigma, gamma):
        value = - self.likelihood(i, mu[i], sigma[i], mu, sigma, gamma) \
                + np.log(sigma[i] * np.sqrt(2*np.pi)) \
                + 1/2 \
                - np.log(self.b0 / self.a0) \
                + self.expected_log_prior(i, mu[i], sigma[i], mu, sigma, gamma)
        return value

    def update_mu(self, i, mu, sigma, gamma):
        return minimize(self.mu_function(i, mu, sigma, gamma), mu[i])

    def update_sigma(self, i, mu, sigma, gamma):
        return minimize(self.sigma_function(i, mu, sigma, gamma), sigma[i], bounds=[(1e-5, np.inf)])

    def update_gamma(self, i, mu, sigma, gamma):
        return logit_inv(self.gamma_function(i, mu , sigma, gamma))

    def estimate_vb_parameters(self, tolerance=1e-5, verbose=False):

        mu, sigma, gamma = self.initial_values()
        gamma_old = gamma.copy()
        
        delta_h = 1
        # Does this need to be updated each time?
        a = np.argsort(mu)[::-1]
        
        start_time = time.time()
        epochs = 0
        while delta_h >= tolerance:
            print(epochs, round(delta_h, 7), round(time.time() - start_time, 0))
            for i in a:
                # Use old values throughout or newest as possible?
                sigma[i] = self.update_sigma(i, mu, sigma, gamma)
                mu[i] = self.update_mu(i, mu, sigma, gamma)
                gamma_old[i] = gamma[i]
                gamma[i] = self.update_gamma(i, mu, sigma, gamma)
                
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

    def expected_log_prior(self, i, mu_i, sigma_i, mu, sigma, gamma) :
        return -(mu_i**2 + sigma_i**2)/2 - np.log(np.sqrt(2*np.pi))

    def update_mu(self, i, mu, sigma, gamma):
        mask = (np.arange(self.p) != i)
        return sigma[i]**2 * (self.YX[i] - (self.XX[i, :] * gamma * mu * mask).sum())

    def update_sigma(self, i, mu, sigma, gamma):
        return 1 / np.sqrt(self.XX[i, i] + 1)

    def __repr__(self):
        return "Gaussian()"


class LaplaceVB(BaseVB):

    def __init__(self, data, lambd=1):
        self.lambd = lambd
        super().__init__(data)

    def expected_log_prior(self, i, mu_i, sigma_i, mu, sigma, gamma):
        expected_abs = sigma_i * np.sqrt(2 / np.pi) * np.exp(- mu_i**2 / (2*sigma_i**2)) \
                       + mu_i * (1 - 2*sp.stats.norm.cdf(-mu_i / sigma_i))
        value = -self.lambd*expected_abs + np.log(self.lambd / 2)
        return value

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

