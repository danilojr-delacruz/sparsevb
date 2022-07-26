import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

directory = "/home/delacruz-danilojr/delacruz/Projects/urop/sparsevb/simulations"

labels = ["Beginning", "Middle", "End", "Uniform"]
p=200
n=100
s=20

class Analyse:

    def __init__(self, name=None, distribution=None, r=None):

        if name is None:
            if distribution == "FatLaplace":
                name = f"{distribution} r={r}"

                self.distribution = distribution
                self.r = r
        else:
            self.name = name
            self.distribution = name

        self.path = f"{directory}/{name}"
        self.summary_df = pd.read_csv(f"{self.path}/summary.csv", index_col=0)

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

