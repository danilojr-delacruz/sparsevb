import sparsevb.vb as vb
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt
import time
import os
import shutil

from sparsevb import *
from pandas.plotting import table

# Specification of simulation
name = input("Specify Simulation Name: ")
directory = f"/home/delacruz-danilojr/delacruz/Projects/urop/sparsevb/simulations/{name}"

if os.path.exists(directory):
    shutil.rmtree(directory)

os.mkdir(directory)
os.mkdir(f"{directory}/figures")
os.mkdir(f"{directory}/data")

# Specification of prior
prior = input("Specify Prior Distribution (Laplace/FatLaplace): ")

if prior == "Laplace":
    vb_class = vb.LaplaceVB
elif prior == "FatLaplace":
    vb_class = vb.FatLaplaceVB
else:
    raise Exception("Prior Distribution not recognised")


# Generation of data
seed = 0

n = 100
p = 200
s = 20

np.random.seed(seed)

labels = ["Beginning", "Middle", "End", "Uniform"]

X = np.random.normal(size=(n, p))
random_non_zero_locations = np.random.choice(range(p), size=s, replace=False)

## Generate the different types of true thetas
theta = np.zeros(shape=(4, p))
### Beginning
theta[0, :s] = 10
### Middle
theta[1, p//2 - s//2: p//2 + s//2] = 10
### End
theta[2, -s:] = 10
### Uniform
theta[3, random_non_zero_locations] = 10

data = [
    generate_data(X, theta[0, :]), generate_data(X, theta[1, :]),
    generate_data(X, theta[2, :]), generate_data(X, theta[3, :])
]

summary = {}

## Posterior Mean
fig_pm, ax_pm = plt.subplots(2, 2, figsize=(20, 10))
fig_pm.tight_layout(pad=5)
fig_pm.suptitle("Posterior Mean", size=25)

## Gamma Activation
fig_gamma, ax_gamma = plt.subplots(2, 2, figsize=(20, 10))
fig_gamma.tight_layout(pad=5)
fig_gamma.suptitle("Gamma Activation", size=25)

parameters = {}
for t in range(4):
    VB = vb_class(data[t])

    start_time = time.time()
    mu, sigma, gamma = VB.estimate_vb_parameters()
    end_time = time.time()
    run_time = end_time - start_time

    posterior_mean = VB.posterior_mean(mu, gamma)

    row, col = t//2, t%2
    ax_pm[row, col].scatter(np.arange(p), posterior_mean, label="VB")
    ax_pm[row, col].scatter(np.arange(p), theta[t], label="Signal")
    ax_pm[row, col].set_title(labels[t], size=20)
    ax_pm[row, col].set_xlabel("Index", size=15)
    ax_pm[row, col].set_ylabel("Signal Value", size=15)

    ax_gamma[row, col].scatter(np.arange(p), gamma, label="VB")
    ax_gamma[row, col].scatter(np.arange(p), theta[t] != 0, label="Signal")
    ax_gamma[row, col].set_title(labels[t], size=20)
    ax_gamma[row, col].set_xlabel("Index", size=15)
    ax_gamma[row, col].set_ylabel("Activation", size=15)

    l2_error = l2(posterior_mean, theta[t])
    fdr, tpr = fdr_tpr(gamma > 0.5, theta[t] != 0)

    summary[labels[t]] = [l2_error, fdr, tpr, run_time]
    parameters[t] = pd.DataFrame({"mu": mu, "sigma": sigma, "gamma": gamma})

## Create Summary
summary_df = pd.DataFrame(summary, index=["$l_2$-error", "FDR", "TPR", "runtime (sec)"])
fig_df, ax_df = plt.subplots()
ax_df.axis("off")
table(ax_df, summary_df.round(4))


### Only want legend to show for first one
ax_pm[0, 0].legend(fontsize=15)
ax_gamma[0, 0].legend(fontsize=15)

# Save pictures
fig_pm.savefig(f"{directory}/figures/posterior_mean.png")
fig_gamma.savefig(f"{directory}/figures/gamma_activation.png")
fig_df.savefig(f"{directory}/figures/summary.png", bbox_inches="tight")

# Save data
summary_df.to_csv(f"{directory}/summary.csv")

for t in range(4):
    prefix = f"{directory}/data/{labels[t]}"
    X, Y = data[t]
    np.savetxt(f"{prefix}_X.csv", X)
    np.savetxt(f"{prefix}_Y.csv", Y)
    np.savetxt(f"{prefix}_theta.csv", theta[t])
    
    parameters[t].to_csv(f"{prefix}_parameters.csv")




