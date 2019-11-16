'''
Plot the size distribution statistics in 3 different representations.
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import os, glob
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pathlib import Path
from tqdm import tqdm

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

# Local libraries
import lib.plot as plib

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def get_counts(df, vol=False, return_df=False):
    df['z'] = df.coord // (1536 * 512)
    df_ = df.groupby(['cid', 'z']).size().reset_index(name='counts')

    if return_df:
        return df_
    elif vol:
        return df.cid.value_counts().values

    return df_.counts.value_counts().values

def get_rank_rep(df):
    counts = get_counts(df)
    x, y = np.unique(counts, return_index=True)

    return x, y

def get_extrema(pq_list):
    maxima = np.zeros(4) # x1, y1, x2, y2
    def _compswap(maxima, index):
        pass

    for time, item in enumerate(tqdm(pq_list)):
        df = pq.read_pandas(item).to_pandas()

        # Rank statistics
        counts = get_rank_rep(df)
        maxima[0] = np.max([])

        # Probability density
        x_grid = np.log10(np.logspace(0, np.log10(np.max(counts)), 35))
        log_kde = KernelDensity(bandwidth=1).fit(np.log(counts[:, None]))
        kde = log_kde.score_samples(x_grid[:, None])


def plot_scatter(time, df, init=False):
    """
    Plot the size distribution of cloud volumes.

    """
    # Select condensed region
    df = df[df.type == 0]

    #---- Plot
    fig = plib.init_plot(figsize=(9, 6))

    # Plot 1: Vertical distribution
    ax1 = fig.add_subplot(221)

    ax1.title.set_text('Vertical Size Distribution')
    ax1.set_xlabel(r'Size')
    ax1.set_ylabel(r'Altitude')
    
    # 3D cloud volumes
    # df['z'] = df.coord // (1536 * 512)
    # grp_ = df.groupby(['cid'])
    # df_ = grp_.z.agg([np.size, np.mean]).reset_index()
    # ax1.scatter(df_['size'], df_['mean'], marker='.')

    # 2D slices, contiguous volumes
    # Always more informative to to is this way
    df_ = get_counts(df, return_df=True)
    ax1.scatter(df_.counts, df_.z, marker='.')

    # Plot 2: Rank-size representation
    ax2 = fig.add_subplot(222)

    ax2.title.set_text('Rank-size Representation')
    ax2.set_xlabel(r'$\log_{10}$ R')
    ax2.set_ylabel(r'$\log_{10}$ S')

    x, y = get_rank_rep(df)
    ax2.scatter(np.log10(x), np.log10(y), marker='.')

    # Plot 3: Cumulative representation
    ax3 = fig.add_subplot(223)

    ax3.title.set_text('Cumulative Represenation')
    ax3.set_xlabel(r'$\log_{10}$ R')
    ax3.set_ylabel(r'$\log_{10}$ S$_{c}$')
    
    x, y = get_rank_rep(df)

    cy = np.cumsum(y[::-1])[::-1]
    y = cy  / cy[0] * 100

    ax3.scatter(np.log10(x), np.log10(y), marker='.')

    # Plot 4: PDF representation
    ax3 = fig.add_subplot(224)

    ax3.title.set_text('Probability Density Representation')
    ax3.set_xlabel(r'$\log_{10}$ Cloud Size')
    ax3.set_ylabel(r'KDE')

    x, y = get_rank_rep(df)
    counts = get_counts(df)

    x_grid = np.log10(np.logspace(0, np.log10(np.max(counts)), 35))
    if init:
        grid = GridSearchCV(
                    KernelDensity(),
                    param_grid = {
                        'bandwidth': np.linspace(0, 5, 200)},
                    n_jobs=16)
        grid.fit(np.log10(x)[:, None])
        print(grid.best_params_)
        bw = grid.best_params_['bandwidth']
    bw = 1 # CV result

    log_kde = KernelDensity(bandwidth=bw).fit(np.log(counts[:, None]))
    kde = log_kde.score_samples(x_grid[:, None])

    ax3.plot(x_grid, kde, '-+', alpha=0.5)
    
    #---- Save figure
    plt.tight_layout()
    file_name = f"../png/{os.path.splitext(__file__)[0]}.png"
    plib.save_fig(fig, file_name)
    # file_name = f"../png/frame/vol_rep_{time:04d}.png"
    # plib.save_fig(fig, file_name, print_output=False)
    # tqdm.write(f"\t Wrote figure to {file_name}")

if __name__ == '__main__':
    src = Path("/users/loh/nodeSSD/temp")

    pq_list = sorted(src.glob('BOMEX_SWAMP_volume_*.pq'))
    # get_extrema(pq_list)
    plot_scatter(719, pq.read_pandas(pq_list[600]).to_pandas())
    # for t, item in enumerate(tqdm(pq_list)):
    #     plot_scatter(t, pq.read_pandas(item).to_pandas())
