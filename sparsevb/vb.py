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
        self.YY = Y.T @ Y
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

    def expected_log_prior(self, mu_i, sigma_i):
        return 0

    def mu_function(self, i, mu, sigma, gamma):
        def func(mu_i):
            value = self.likelihood(i, mu_i, sigma[i], mu, sigma, gamma) \
                    - self.expected_log_prior(mu_i, sigma[i])
            return value
        return func

    def sigma_function(self, i, mu, sigma, gamma):
        def func(sigma_i):
            value = self.likelihood(i, mu[i], sigma_i, mu, sigma, gamma) \
                    - np.log(sigma_i) \
                    - self.expected_log_prior(mu[i], sigma_i)
            return value
        return func

    def gamma_function(self, i, mu, sigma, gamma):
        value = - self.likelihood(i, mu[i], sigma[i], mu, sigma, gamma) \
                + np.log(sigma[i] * np.sqrt(2*np.pi)) \
                + 1/2 \
                - np.log(self.b0 / self.a0) \
                + self.expected_log_prior(mu[i], sigma[i])
        return value

    def update_mu(self, i, mu, sigma, gamma):
        return minimize(self.mu_function(i, mu, sigma, gamma), mu[i])

    def update_sigma(self, i, mu, sigma, gamma):
        return minimize(self.sigma_function(i, mu, sigma, gamma), sigma[i], bounds=[(1e-5, np.inf)])

    def update_gamma(self, i, mu, sigma, gamma):
        return logit_inv(self.gamma_function(i, mu , sigma, gamma))

    def estimate_vb_parameters(self, tolerance=1e-5, verbose=False, min_epochs=10, max_epochs=1000,
            max_patience=10, patience_factor=2):
        """Patience factor means to increase patience next delta_h needs to be within
        patience_factor*tolerance of the previous one"""

        mu, sigma, gamma = self.initial_values()
        gamma_old = gamma.copy()
        
        delta_h = 1
        a = np.argsort(mu)[::-1]
        
        start_time = time.time()
        epochs = 0
        patience = 0 # If reaches max patience we stop

        while (delta_h > tolerance) | (epochs < min_epochs) | (patience < max_patience):

            if epochs >= max_epochs:
                if verbose:
                    print("Convergence failed")
                break

            for i in a:
                mu[i] = self.update_mu(i, mu, sigma, gamma)
                sigma[i] = self.update_sigma(i, mu, sigma, gamma)
                gamma_old[i] = gamma[i]
                gamma[i] = self.update_gamma(i, mu, sigma, gamma)
                
            new_delta_h = DeltaH(gamma_old, gamma)
            if abs(delta_h - new_delta_h) <= patience_factor*tolerance:
                patience += 1
            else:
                patience = 0

            delta_h = new_delta_h
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

    def nelbo(self, mu, sigma, gamma):
        XXggmm = self.XX * np.outer(gamma*mu, gamma*mu)
        likelihood = self.YY \
                    -2 * self.YX @ (gamma*mu) \
                    + XXggmm.sum() - XXggmm.trace() \
                    + np.diag(self.XX) @ (gamma * (mu**2 + sigma**2))
        likelihood /= 2
        
        w_bar = self.a0 / (self.a0 + self.b0)
        term_1 = np.log(gamma / w_bar) - np.log(sigma * np.sqrt(2*np.pi)) \
                - 1/2 - self.expected_log_prior(mu, sigma)
        term_2 = np.log(1 - gamma) - np.log(1 - w_bar)
        
        variational_and_prior = gamma*term_1 + (1-gamma)*term_2
        variational_and_prior = variational_and_prior.sum()
        
        return likelihood + variational_and_prior

    def elbo(self, mu, sigma, gamma):
        return -self.nelbo(mu, sigma, gamma)

    def get_history(self, tolerance=1e-5, verbose=False, min_epochs=10, max_epochs=1000,
            max_patience=10, patience_factor=2):
        """Each row represent history of parameter"""

        mu, sigma, gamma = self.initial_values()
        gamma_old = gamma.copy()
        
        delta_h = 1
        a = np.argsort(mu)[::-1]

        history = {"mu": [mu.copy()], "sigma": [sigma.copy()],
                "gamma": [gamma.copy()], "delta_h": [delta_h]}
 
        start_time = time.time()
        epochs = 0
        patience = 0 # If reaches max patience we stop

        while (delta_h > tolerance) | (epochs < min_epochs) | (patience < max_patience):
            if epochs >= max_epochs:
                if verbose:
                    print("Convergence failed")
                break

            for i in a:
                mu[i] = self.update_mu(i, mu, sigma, gamma)
                sigma[i] = self.update_sigma(i, mu, sigma, gamma)
                gamma_old[i] = gamma[i]
                gamma[i] = self.update_gamma(i, mu, sigma, gamma)
                
            new_delta_h = DeltaH(gamma_old, gamma)
            if abs(delta_h - new_delta_h) <= patience_factor*tolerance:
                patience += 1
            else:
                patience = 0

            delta_h = new_delta_h
            epochs += 1

            history["mu"].append(mu.copy())
            history["sigma"].append(sigma.copy())
            history["gamma"].append(gamma.copy())
            history["delta_h"].append(delta_h)

        end_time = time.time()
        run_time = end_time - start_time

        if verbose:
            print(f"Ran {epochs} epochs in {round(run_time, 4)} seconds.")
            print(f"Final change in binary maximal entropy is {round(delta_h, 5)}.")
        
        history["mu"] = np.vstack(history["mu"]).T
        history["sigma"] = np.vstack(history["sigma"]).T
        history["gamma"] = np.vstack(history["gamma"]).T
        history["delta_h"] = np.array(history["delta_h"])

        return history

    def __repr__(self):
        return "BaseVB()"

    def __str__(self):
        return self.__repr()


class GaussianVB(BaseVB):

    def expected_log_prior(self, mu_i, sigma_i):
        return -(mu_i**2 + sigma_i**2)/2 - np.log(np.sqrt(2*np.pi))

    def update_mu(self, i, mu, sigma, gamma):
        mask = (np.arange(self.p) != i)
        sigma_i = self.update_sigma(i, mu, sigma, gamma)
        return sigma_i**2 * (self.YX[i] - (self.XX[i, :] * gamma * mu * mask).sum())
        

    def update_sigma(self, i, mu, sigma, gamma):
        return 1 / np.sqrt(self.XX[i, i] + 1)

    def __repr__(self):
        return "Gaussian()"


class LaplaceVB(BaseVB):

    def __init__(self, data, lambd=1):
        self.lambd = lambd
        super().__init__(data)

    def expected_log_prior(self, mu_i, sigma_i):
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

    def expected_log_prior(self, mu_i, sigma_i):
        r = self.r
        coeff = (r * self.lambd**(1/r)) / (2 * sp.special.gamma(1/r))
        rth_moment = (sigma_i**r) * (2**(r/2)) \
                     * sp.special.gamma((r+1) / 2) / sp.special.gamma(1/2) \
                     * sp.special.hyp1f1(-r/2, 1/2, -1/2 * (mu_i / sigma_i)**2)
        return np.log(coeff) - self.lambd*rth_moment

    def __repr__(self):
        return f"FatLaplace(lambd={self.lambd}, r={self.r})"


class JensenBoundCauchy(BaseVB):

    # Apply JensenBound instead of true value.
    def expected_log_prior(self, mu_i, sigma_i):
        value = - np.log(1 + mu_i**2 + sigma_i**2) \
                - np.log(np.pi)
        return value

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

    def expected_log_prior(self, mu_i, sigma_i):
        return -self.neg_expected_log_g(mu_i, sigma_i)

    def __repr__(self):
        return "NumericIntCauchy()"

