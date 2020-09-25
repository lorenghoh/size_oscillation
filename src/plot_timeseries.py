import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq 
import seaborn as sns

from pathlib import Path

import lib.plot as plib


def main():
    case = 'BOMEX_SWAMP' # Case name 
    c_type = 'cloud'

    #---- Plotting 
    fig = plib.init_plot(figsize=(10, 4))
    ax = fig.add_subplot(111)
    
    x = np.arange(0, 720)

    # Plot timeseries
    file_name = f'pq/size_dist_slope.pq'
    df = pq.read_pandas(file_name).to_pandas()

    ax.plot(x, -df.lr[:720], 'k-', lw=0.75,
            label='Linear Regression')
    ax.plot(x, -df.rs[:720], '-*', lw=0.75,
            label='RANSAC Regression')
    ax.plot(x, -df.ts[:720], '--', lw=0.75,
            label='Theil-Sen Regression')

    ax.set_xlabel(r'Time [min]')
    ax.set_ylabel(r'Slope $-b$')

    ax.legend()

#     # # Vars on the second scale
#     file_name = f'../pq/{case}_vars.pq'
#     df2 = pq.read_pandas(file_name).to_pandas()

#     ax2 = ax.twinx()
#     ax2.plot()

#     ax2.plot(x, df2.qp_ave, 'k-', lw=1.25,
#         label="Ave. Precip. [g/kg]"    
#     )

#     ax2.set_xticklabels([i * 60 for i in x])
#     ax2.set_ylabel(r'Ave. Precip. [g/kg]')
    
#     lines, labels = ax.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     ax2.legend(lines + lines2, labels + labels2)

    plib.save_fig(fig, Path(__file__))


if __name__ == '__main__':
    main()
