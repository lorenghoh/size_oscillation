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
    case = 'BOMEX_BOWL' # Case name 
    c_type = 'cloud'

    df = pq.read_pandas(f'../pq/{case}_{c_type}_size_dist_slope.pq')
    df = df.to_pandas()
    
    x = np.arange(0, 540)

    #---- Plotting 
    fig = plt.figure(1, figsize=(16, 4))
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
    
    plt.plot(x, df.lr, 'k-', lw=0.75,
            label='Linear Regression')
    plt.plot(x, df.ts, '-*', lw=0.75,
            label='Theil-Sen Regression')
    plt.plot(x, df.hb, '--', lw=0.75,
            label='Huber Regression')

    plt.xlabel(r'Time [min]')
    plt.ylabel(r'Slope $b$')

    plt.legend()
    
    plt.tight_layout(pad=0.5)
    figfile = '../png/{}.png'.format(os.path.splitext(__file__)[0])
    print('\t Writing figure to {}...'.format(figfile))
    plt.savefig(figfile,bbox_inches='tight', dpi=180, \
                facecolor='w', transparent=True)

if __name__ == '__main__':
    main()