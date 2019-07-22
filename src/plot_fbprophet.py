import os, glob

import numpy as np
import numba as nb
import pandas as pd
import pyarrow.parquet as pq

from fbprophet import Prophet

import lib_plot as plib
import seaborn as sns

def main():
    df = pq.read_pandas('../pq/BOMEX_12HR_cloud_2d_size_dist_slope.pq').to_pandas()
    t_stamps = pd.date_range("00:00", freq="D", periods=540)

    ds = pd.DataFrame({
        'ds': t_stamps,
        'y': df.rs.astype(np.float_)
    })

    # Set model
    m = Prophet()
    m.add_seasonality(name='Perodicity', period=45, fourier_order=5)
    m.fit(ds)
    
    # Forecast
    fcast = m.predict(m.make_future_dataframe(periods=120))

    #---- Plotting 
    # fig = m.plot_components(fcast)
    fig = m.plot(fcast)
    ax = fig.add_subplot(111)
    
    file_name = f'../png/plot_fbprophet.png'
    plib.save_fig(fig, file_name, False)

if __name__ == '__main__':
    main()