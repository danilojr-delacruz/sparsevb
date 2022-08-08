import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats
import os
import matplotlib.pyplot as plt

from sparsevb import l2

directory = "/home/delacruz-danilojr/delacruz/Projects/urop/sparsevb/simulations"

labels = ["Beginning", "Middle", "End", "Uniform"]
p=200
n=100
s=20


def credible_region(mu, sigma, gamma, alpha=0.05):

    cdf = sp.stats.norm.cdf
    ppf = sp.stats.norm.ppf

    if gamma <= alpha:
        lower, upper = 0, 0
    else:
        f = mu + sigma*ppf(cdf(-mu/sigma) + np.sign(mu)*(1 - alpha/gamma))

        if max(cdf(-mu/sigma), cdf(mu/sigma)) <= 1 - alpha/(2*gamma):
            multiplier = ppf(1 - alpha/(2*gamma))
            lower = mu - sigma*multiplier
            upper = mu + sigma*multiplier
        elif gamma < 1 - alpha:
            lower, upper = sorted([0, f])
        else:
            multiplier = ppf(1/2 + (1-alpha)/(2*gamma))
            cand_lower = mu - sigma*multiplier
            cand_upper = mu + sigma*multiplier
            
            if (cand_lower <= 0) and (0 <= cand_upper):
                lower, upper = sorted([0, f])
            else:
                option_1_length = abs(f)
                option_2_length = 2*sigma*multiplier

                if option_1_length <= option_2_length:
                    lower, upper = sorted([0, f])
                else:
                    lower = cand_lower
                    upper = cand_upper

    return lower, upper

credible_region = np.vectorize(credible_region, otypes=[float, float])




class Analyse:

    def __init__(self, name=None, distribution=None, r=None, design_matrix="iid_elements"):

        if name is None:
            if distribution == "FatLaplace":
                name = f"{distribution} r={r}"

                self.distribution = distribution
                self.r = r
        else:
            self.name = name
            self.distribution = name

        self.path = f"{directory}/{design_matrix}/{name}"
        self.summary_df = pd.read_csv(f"{self.path}/summary.csv", index_col=0)

    def get_data(self, name, label="Beginning"):
        """label one of [Beginning, Middle, End, Uniform].
        name one of [index, parameters, theta, X, Y]
        """
        path = f"{self.path}/data/{label}_{name}.csv"
        if name == "parameters":
            df = pd.read_csv(path, index_col=0)
            arr = np.array(df)
            mu, sigma, gamma = arr[:, 0], arr[:, 1], arr[:, 2]
            return mu, sigma, gamma
        if name == "index":
            theta = self.get_data("theta", label=label)
            pos_index = np.where(theta != 0)[0]
            neg_index = np.where(theta == 0)[0]
            all_index = np.arange(p)
            return pos_index, neg_index, all_index
        else:
            arr = np.loadtxt(path)
            return arr

    @staticmethod
    def credible_region(mu, sigma, gamma, alpha=0.05):
        return credible_region(mu, sigma, gamma, alpha)

    def credible_region_summary(self, label, mode, alpha):
        mu, sigma, gamma = self.get_data("parameters", label=label)
        theta = self.get_data("theta", label=label)

        lower, upper = self.credible_region(mu, sigma, gamma, alpha)
        posterior_mean = mu*gamma

        pos_index = np.where(theta != 0)[0]
        neg_index = np.where(theta == 0)[0]
        all_index = np.arange(p)

        success = (lower <= theta) & (theta <= upper)
        cr_length = upper - lower
        gamma_binarity = np.minimum(1-gamma, gamma)

        if mode == "positive":
            index = pos_index
        elif mode == "negative":
            index = neg_index
        else:
            index = all_index

        return success[index].mean(), cr_length[index].mean(), gamma_binarity[index].mean()

    def credible_region_summary_df(self, label="Beginning", alphas=[0.1, 0.05, 0.01]):
        modes = ["positive", "negative", "overall"]
        result = {mode: {} for mode in modes}
        for alpha in alphas:
            for mode in modes:
                accuracy, length, binarity = self.credible_region_summary(label, mode, alpha)
                result[mode][alpha] = {}
                result[mode][alpha]["accuracy"] = accuracy
                result[mode][alpha]["average cr_length"] = length
                result[mode][alpha]["average gamma_binarity"] = binarity

        df = pd.concat([pd.DataFrame(result[mode]) for mode in modes], keys=modes)
        df = df.swaplevel().sort_index()

        return df

    @staticmethod
    def l2_error(x, y):
        return l2(x, y)

    @staticmethod
    def gamma_binarity(gamma):
        return np.minimum(1-gamma, gamma).mean()

    @staticmethod
    def credible_region_accuracy(theta, lower, upper):
        return ((lower <= theta) & (theta <= upper)).mean()

    @staticmethod
    def credible_region_average_length(lower, upper):
        return (upper - lower).mean()

    def plot_credible_regions(self, label="Beginning", mode="positive", alpha=0.05, ax=None, true_param_markersize=15):
        """mode one of [positive, negative, both]
        true_param_markersize controls markersize of true parameter.
        """

        mu, sigma, gamma = self.get_data("parameters", label=label)
        theta = self.get_data("theta", label=label)
        
        lower, upper = self.credible_region(mu, sigma, gamma, alpha)
        posterior_mean = mu*gamma

        pos_index = np.where(theta != 0)[0]
        neg_index = np.where(theta == 0)[0]

        if mode == "positive":
            index = pos_index
        elif mode == "negative":
            index = neg_index
        else:
            index = np.arange(p)
            fig, ax = plt.subplots(ncols=2, figsize=(20, 5))
            self.plot_credible_regions(label=label, mode="positive", alpha=alpha, ax=ax[0])
            self.plot_credible_regions(label=label, mode="negative", alpha=alpha, ax=ax[1])
            return

        if ax is None:
            fig, ax = plt.subplots(1, figsize=(20, 5))
 
        x = np.arange(len(index))
        ax.fill_between(x, lower[index], upper[index], label="Credible Interval")
        ax.scatter(x, posterior_mean[index], label="Posterior Mean")
        ax.plot(x, theta[index], "v", markersize=true_param_markersize, label="True Parameter", color="tab:green")
        ax.set_xticks(x[::len(index)//20], index[::len(index)//20])
        ax.set_title(mode, size=20)
        ax.legend()

    def plot_posterior_mean(self):
        fig, ax = plt.subplots(2, 2, figsize=(20, 10))
        fig.tight_layout(pad=5)
        fig.suptitle("Posterior Mean", size=25)

        for t in range(4):
            label = labels[t]
            parameter_df = pd.read_csv(f"{self.path}/data/{label}_parameters.csv", index_col=0)
            arr = np.array(parameter_df)
            mu, sigma, gamma = arr[:, 0], arr[:, 1], arr[:, 2]
            posterior_mean = mu * gamma

            theta = np.loadtxt(f"{self.path}/data/{label}_theta.csv")

            row, col = t//2, t%2
            ax[row, col].scatter(np.arange(p), posterior_mean, label="VB")
            ax[row, col].scatter(np.arange(p), theta, label="Signal")
            ax[row, col].set_title(label, size=20)
            ax[row, col].set_xlabel("Index", size=15)
            ax[row, col].set_ylabel("Signal Value", size=15)

        ax[0, 0].legend(fontsize=15)

        return fig, ax
        
    def plot_gamma_activation(self):
        fig, ax = plt.subplots(2, 2, figsize=(20, 10))
        fig.tight_layout(pad=5)
        fig.suptitle("Gamma Activation", size=25)

        for t in range(4):
            label = labels[t]
            parameter_df = pd.read_csv(f"{self.path}/data/{label}_parameters.csv", index_col=0)
            arr = np.array(parameter_df)
            mu, sigma, gamma = arr[:, 0], arr[:, 1], arr[:, 2]

            theta = np.loadtxt(f"{self.path}/data/{label}_theta.csv")

            row, col = t//2, t%2
            ax[row, col].scatter(np.arange(p), gamma, label="VB")
            ax[row, col].scatter(np.arange(p), theta != 0, label="Signal")
            ax[row, col].set_title(label, size=20)
            ax[row, col].set_xlabel("Index", size=15)
            ax[row, col].set_ylabel("Activation", size=15)

        ax[0, 0].legend(fontsize=15)

        return fig, ax


class LaplaceCalibration:

    def __init__(self):

        self.r_values = sorted([float(folder[13:]) for folder in os.listdir(directory) if "FatLaplace" in folder])
        # Exclude really small r
        self.r_values = self.r_values[1:]
        data = {}
        for r in self.r_values:
            path = f"{directory}/FatLaplace r={r}/summary.csv"
            df = pd.read_csv(path, index_col=0)
            data[r] = df
        self.joint_df = pd.concat([data[r] for r in self.r_values], keys=self.r_values)

        self.laplace_df = pd.read_csv(f"{directory}/Laplace/summary.csv", index_col=0)
        self.gaussian_df = pd.read_csv(f"{directory}/Gaussian/summary.csv", index_col=0)

    def plot_FatLaplace_calibration():
        pass

    def plot_FatLaplace_vs_Laplace(self, theta_loc=labels[0]):
        fig, ax = plt.subplots(2, 2, figsize=(20, 10))
        fig.tight_layout(pad=5)
        for i, metric in enumerate(self.joint_df.index.unique(level=1)):
            row, col = i//2, i%2
            ax[row, col].plot(self.r_values, self.joint_df.loc[:, metric, :][theta_loc], label="Fat Laplace")
            ax[row, col].scatter(self.r_values, self.joint_df.loc[:, metric, :][theta_loc], label="Fat Laplace")
            ax[row, col].axhline(self.laplace_df.loc[metric, theta_loc], ls="--", label="Laplace", color="tab:green")
            ax[row, col].set_xlabel("$p$", size=15)
            ax[row, col].set_ylabel("Value", size=15)
            ax[row, col].set_title(metric, size=20)
            
        fig.suptitle(f"Metrics of FatLaplace as $r$ is varied compared against Laplace ({theta_loc})", size=20)
        ax[0, 0].legend(fontsize=15)

    def plot_FatLaplace_vs_Gaussian(self, theta_loc=labels[0]):
        fig, ax = plt.subplots(2, 2, figsize=(20, 10))
        fig.tight_layout(pad=5)
        for i, metric in enumerate(self.joint_df.index.unique(level=1)):
            row, col = i//2, i%2
            ax[row, col].plot(self.r_values, self.joint_df.loc[:, metric, :][theta_loc], label="Fat Laplace")
            ax[row, col].scatter(self.r_values, self.joint_df.loc[:, metric, :][theta_loc], label="Fat Laplace")
            ax[row, col].axhline(self.gaussian_df.loc[metric, theta_loc], ls="--", label="Gaussian", color="tab:green")
            ax[row, col].set_xlabel("$p$", size=15)
            ax[row, col].set_ylabel("Value", size=15)
            ax[row, col].set_title(metric, size=20)
            
        fig.suptitle(f"Metrics of FatLaplace as $r$ is varied compared against Gaussian ({theta_loc})", size=20)
        ax[0, 0].legend(fontsize=15)

