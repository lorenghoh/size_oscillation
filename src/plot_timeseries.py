import os, glob

import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    df = pq.read_pandas('../pq/cloud_size_dist_slope.pq').to_pandas()

    x = np.arange(0, 540)
    y = df.slope

    #---- Plotting 
    fig = plt.figure(1, figsize=(10, 4))
    fig.clf()
    sns.set_context('paper')
    sns.set_style('ticks', 
        {
            'axes.grid': False,
            'axes.linewidth': '0.75',
            'grid.color': '0.75',
            'grid.linestyle': u':',
            'legend.frameon': True,
        })
    plt.rc('text', usetex=True)
    plt.rc('font', family='Serif')

    ax = plt.subplot(1, 1, 1)
    
    plt.plot(x, y, '-*', lw=0.75)

    plt.xlabel(r'Time [min]')
    plt.ylabel(r'Slope $b$')
    
    plt.tight_layout(pad=0.5)
    figfile = '../png/{}.png'.format(os.path.splitext(__file__)[0])
    print('\t Writing figure to {}...'.format(figfile))
    plt.savefig(figfile,bbox_inches='tight', dpi=180, \
                facecolor='w', transparent=True)

if __name__ == '__main__':
    main()