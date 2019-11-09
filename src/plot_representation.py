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

# Local libraries
import lib.plot as plib

def plot_scatter(df):
    """
    Plot the size distribution of cloud volumes.
    
    """
    # Select condensed region
    df = df[df.type == 0]

    #---- Plot
    fig = plib.init_plot(figsize=(8, 6))

    # Plot 1: Raw distribution
    ax1 = fig.add_subplot(221)

    ax1.set_xlabel(r'R')
    ax1.set_ylabel(r'S')

    x = df.cid.value_counts().values
    y = df.cid.value_counts().index.to_numpy()

    ax1.scatter(np.log10(x), np.log10(y), marker='.')

    # Plot 1: Rank-size representation
    ax2 = fig.add_subplot(222)

    ax2.set_xlabel(r'R')
    ax2.set_ylabel(r'S')

    sizes = df.cid.value_counts().values
    x, y = np.unique(sizes, return_index=True)

    ax2.scatter(np.log10(x), np.log10(y), marker='.')

    # Plot 2: Cumulative representation
    ax3 = fig.add_subplot(223)

    ax3.set_xlabel(r'R')
    ax3.set_ylabel(r'S')
    
    cy = np.cumsum(y[::-1])[::-1]
    y = cy  / cy[0] * 100

    ax3.scatter(np.log10(x), np.log10(y), marker='.')
    
    #---- Save figure
    plt.tight_layout()
    file_name = f"../png/{os.path.splitext(__file__)[0]}.png"
    plib.save_fig(fig, file_name)

if __name__ == '__main__':
    src = Path("/users/loh/nodeSSD/temp")

    pq_list = sorted(src.glob('BOMEX_SWAMP_volume_*.pq'))
    plot_scatter(pq.read_pandas(pq_list[25]).to_pandas())
