import warnings
import numpy as np

from pathlib import Path
import seaborn as sns

from sklearn.neighbors import KernelDensity
from sklearn import linear_model as lm

# Set-up local (and temporary) sys.path for import
# All scripts for calculations and plots need this
from context import add_path
add_path(Path('.').resolve())

try:
    import lib.plot as plib
    import lib.config

    from reg.outliers import detect_outliers
    from reg.samples import cloud_dz as sample
    from reg.distributions import kde as distribution
    from reg.slopes import regressor as reg
except Exception:
    raise Exception("Issue with dynamic import")

config = lib.config.read_config()
pwd = Path(config['pwd'])
src = Path(config['case']) / 'clusters'


def plot_outliers(x, y):
    # ---- Plotting
    fig = plib.init_plot((8, 6))

    for i in range(4):
        ax = fig.add_subplot(2, 2, i + 1)

        # Filter outliers
        try:
            if i == 1:
                regressor = None
            elif i == 2:
                regressor = lm.RANSACRegressor()
            elif i == 3:
                regressor = lm.HuberRegressor()
            else:
                raise ValueError
        except ValueError:
            ax.plot(np.array(x[:, None]), y, "k-*", lw=0.5)
            X_, Y_, = np.array(x), y
        else:
            X_, Y_, mask_ = detect_outliers(regressor, np.array(x), y)

            # Print in/outliers
            ax.plot(np.array(x[:, None]), y[:], "k-", lw=0.5)
            ax.plot(np.array(x[~mask_, None]), y[~mask_], "*", c="k")
            ax.plot(np.array(x[mask_, None]), y[mask_], "*", c="r")

        # RANSAC
        rs_estimator = lm.RANSACRegressor()
        rs_estimator.fit(X_[:, None], Y_)
        y_rs = rs_estimator.estimator_.predict(np.array(x)[:, None])
        ax.plot(np.array(x), y_rs, "-", label="RANSAC")

        # Theil Sen regression
        ts_model = lm.TheilSenRegressor(n_jobs=16)
        ts_model.fit(X_[:, None], Y_)
        y_ts = ts_model.predict(np.array(x)[:, None])
        ax.plot(np.array(x), y_ts, "--", label="Theil-Sen")

        # Huber regression
        hb_model = lm.HuberRegressor()
        hb_model.fit(X_[:, None], Y_)
        y_hb = hb_model.predict(np.array(x)[:, None])
        ax.plot(np.array(x), y_hb, "-.", label="Huber")

        # Labels
        ax.legend()

        if i == 0:
            ax.set_title("No Outliers")
        elif i == 1:
            ax.set_title("Linear Detection")
        elif i == 2:
            ax.set_title("RANSAC Outliers")
        elif i == 3:
            ax.set_title("Huber Outliers")
        else:
            raise ("Subplot id not recognized")

        ax.set_xlabel(r"$\log_{10}$ R")
        ax.set_ylabel(r"$\log_{10}$ S")
        # ax.set_ylim([-10, 1])

    file_name = Path(f"{pwd}/png/outlier_detection.png")
    plib.save_fig(fig, file_name)


def main():
    cluster_list = sorted(src.glob('*.pq'))
    cluster = cluster_list[-1]

    samples = sample(cluster)
    x, y = distribution(samples)

    # mask = (y > -100) 
    # x = x[mask]
    # y = y[mask]

    plot_outliers(x, y)


if __name__ == "__main__":
    main()
