import torch
import torch.nn as nn

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, HMC, Predictive

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pathlib import Path
import seaborn as sns

import lib.plot
import lib.config


config = lib.config.read_config()


def get_timeseries():
    src = Path(config["data"])

    file_name = f"{src}/size_dist_slope.pq"
    reg_model = "lr"  # See get_linear_reg.py for choices

    # file_name = f"{src}/kde_grade_timeseries.pq"
    # reg_model = "min_grad"

    df = pq.read_pandas(file_name).to_pandas()

    x = np.arange(0, 360)
    # np.random.shuffle(x)
    # x = x[:360]

    y = -np.concatenate(df[f"{reg_model}"].ravel())[360:]
    # y = np.array(df[f"{reg_model}"])[x]
    y = (y - np.mean(y)) * 10
    # return torch.arange(720, dtype=torch.double), torch.from_numpy(y)
    return torch.from_numpy(x).double(), torch.from_numpy(y).double()


def plot(
    ax,
    X,
    y,
    plot_observed_data=False,
    plot_predictions=False,
    n_prior_samples=0,
    model=None,
    kernel=None,
    n_test=500,
):

    if plot_observed_data:
        ax.plot(X.numpy(), y.numpy(), "k--*")
    if plot_predictions:
        Xtest = torch.linspace(0, 360, n_test).double()  # test inputs
        # compute predictive mean and variance
        with torch.no_grad():
            mean, cov = model(Xtest, full_cov=True, noiseless=False)
        sd = cov.diag().sqrt()  # standard deviation at each input point x
        ax.plot(Xtest.numpy(), mean.numpy(), "r", lw=2)  # plot the mean
        ax.fill_between(
            Xtest.numpy(),  # plot the two-sigma uncertainty about the mean
            (mean - 2.0 * sd).numpy(),
            (mean + 2.0 * sd).numpy(),
            color="C0",
            alpha=0.3,
        )


def main():
    x, y = get_timeseries()

    # Hygiene
    pyro.clear_param_store()

    kernel = gp.kernels.Periodic(1)
    kernel.lengthscale = pyro.nn.PyroSample(dist.LogNormal(1, 10))
    kernel.period = pyro.nn.PyroSample(dist.LogNormal(1, 10))

    gpr = gp.models.GPRegression(
        x,
        y,
        kernel,
        noise=torch.tensor(0.1),
    )

    # hmc_kernel = NUTS(gpr.model, max_tree_depth=5)
    # mcmc = MCMC(hmc_kernel, num_samples=200, warmup_steps=100)
    # mcmc.run()
    # print(mcmc.summary(prob=0.95))

    optimizer = torch.optim.Adam(gpr.parameters(), lr=0.005)
    loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
    num_steps = 500
    for i in range(num_steps):
        optimizer.zero_grad()
        loss = loss_fn(gpr.model, gpr.guide)
        loss.backward()
        optimizer.step()

    # ---- Plotting
    fig = lib.plot.init_plot(figsize=(12, 4))
    ax = fig.add_subplot(111)

    # Plot labels
    # ax.set_xlabel(r"Period [min]")
    # ax.set_ylabel(r"Slope $b$")

    # sns.distplot(mcmc.get_samples()["kernel.period"])
    plot(ax, x, y, plot_observed_data=True, plot_predictions=True, model=gpr)

    lib.plot.save_fig(fig, Path(__file__))


if __name__ == "__main__":
    main()
