import sparsevb.vb as vb
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import time
import os
import shutil

from sparsevb import *
from sparsevb.analyse import Analyse
from pandas.plotting import table

# Specification of simulation
name = input("Specify Simulation Name: ")

# Specification of prior
prior = input("Specify Prior Distribution (Gaussian/Laplace/FatLaplace/JensenBoundCauchy): ")
r = None

if prior == "Gaussian":
    vb_class = vb.GaussianVB
elif prior == "Laplace":
    vb_class = vb.LaplaceVB
elif prior == "FatLaplace":
    r = float(input("Specify 0 < r < 1: "))
    vb_class = vb.FatLaplaceVB
elif prior == "JensenBoundCauchy":
    vb_class = vb.JensenBoundCauchy
elif prior == "NumericIntCauchy":
    vb_class = vb.NumericIntCauchy
else:
    raise Exception("Prior Distribution not recognised")

# Specification of dataset
design_matrix = input("Specify the Design Matrix: ")

if design_matrix == "iid_elements":
    pass
elif design_matrix == "iid_rows":
    pass
else:
    raise Exception("Design Matrix not recognised.")

directory = f"/home/delacruz-danilojr/delacruz/Projects/urop/sparsevb/simulations/{design_matrix}/{name}"

if os.path.exists(directory):
    shutil.rmtree(directory)

os.mkdir(directory)
os.mkdir(f"{directory}/figures")
os.mkdir(f"{directory}/data")


# Generation of data
seed = 0

n = 100
p = 200
s = 20

np.random.seed(seed)

labels = ["Beginning", "Middle", "End", "Uniform"]

if design_matrix == "iid_elements": 
    X = np.random.normal(size=(n, p))
elif design_matrix == "iid_rows":
    cov = sklearn.datasets.make_spd_matrix(p, random_state=0)
    X = np.random.multivariate_normal(mean=np.zeros(p), cov=cov, size=n)

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

# Perform VB
parameters = {}
run_diagnostics = {}
for t in range(4):
    if prior == "FatLaplace":
        VB = vb_class(data[t], r=r)
    else:
        VB = vb_class(data[t])

    print(labels[t])
    mu, sigma, gamma, epochs, run_time, delta_h = VB.estimate_vb_parameters(verbose=True, include_run_details=True)

    parameters[t] = pd.DataFrame({"mu": mu, "sigma": sigma, "gamma": gamma})
    run_diagnostics[t] = {"epochs": epochs, "run_time (sec)": run_time, "delta_h": delta_h}

# Save Data
for t in range(4):
    prefix = f"{directory}/data/{labels[t]}"
    X, Y = data[t]
    np.savetxt(f"{prefix}_X.csv", X)
    np.savetxt(f"{prefix}_Y.csv", Y)
    np.savetxt(f"{prefix}_theta.csv", theta[t])
    
    parameters[t].to_csv(f"{prefix}_parameters.csv")

# Perform Analysis
analyser = Analyse(name=name, distribution=prior, r=r, design_matrix=design_matrix)

## Create Summary
summary = {}
for t in range(4):
    label = labels[t]
    mu, sigma, gamma = analyser.get_data("parameters", label=label)
    pos_index, neg_index, all_index = analyser.get_data("index", label=label)
    theta = analyser.get_data("theta")
    
    posterior_mean = mu*gamma
    fdr, tpr = fdr_tpr(gamma > 0.5, theta != 0)
    lower, upper = analyser.credible_region(mu, sigma, gamma, alpha=0.05)

    summary[label] = {
        ("FDR", "all"): fdr,
        ("TPR", "all"): tpr,
        ("Epochs", "all"): run_diagnostics[t]["epochs"],
        ("Runtime (sec)", "all"): run_diagnostics[t]["run_time (sec)"],
        ("Delta H", "all"): run_diagnostics[t]["delta_h"],
    }
    
    for descriptor, func, args in [
        ("L2-Error", analyser.l2_error, (posterior_mean, theta)),
        ("Gamma Binarity", analyser.gamma_binarity, (gamma,)),
        ("CR Inclusion Accuracy", analyser.credible_region_accuracy, (theta, lower, upper)),
        ("CR Average Length", analyser.credible_region_average_length, (lower, upper))
    ]:
        for target, index in [
            ("positive", pos_index), ("negative", neg_index), ("all", all_index)
        ]:
            processed_args = [arg[index] for arg in args]
            summary[label][(descriptor, target)] = func(*processed_args)
summary_df = pd.DataFrame(summary)
summary_df.index.names = ("Metric", "Target")

fig_df, ax_df = plt.subplots()
ax_df.axis("off")
table(ax_df, summary_df.round(6))

### Only want legend to show for first one
fig_pm, _ = analyser.plot_posterior_mean()
fig_gamma, _ = analyser.plot_gamma_activation()

## Save pictures
fig_pm.savefig(f"{directory}/figures/posterior_mean.png")
fig_gamma.savefig(f"{directory}/figures/gamma_activation.png")
fig_df.savefig(f"{directory}/figures/summary.png", bbox_inches="tight")

## Save data
summary_df.to_csv(f"{directory}/summary.csv")



