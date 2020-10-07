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
    from reg.samples import get_pq
    from reg.distributions import rank as distribution
    from reg.slopes import regressor as reg
except Exception:
    raise Exception("Issue with dynamic import")

config = lib.config.read_config()
pwd = Path(config['pwd'])
src = Path(config['case']) / 'clusters'


def plot_outliers(df):
    # ---- Plotting
    fig = plib.init_plot((8, 6))

    # mask = (x > 1) & (y > 1)
    # x, y = np.log10(x[mask]), np.log10(y[mask])

    df = df.groupby(['cid', 'z']).size().reset_index(name='counts')

    x = np.logspace(0, np.log10(df.counts.max()), 50)

    log_kde = KernelDensity(bandwidth=10).fit(np.array(df.counts)[:, None])
    y = log_kde.score_samples(x[:, None]) / np.log(10)

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
            ax.plot(np.log10(x[:, None]), y, "k-*", lw=0.5)
            X_, Y_, = np.log10(x), y
        else:
            X_, Y_, mask_ = detect_outliers(regressor, np.log10(x), y)

            # Print in/outliers
            ax.plot(np.log10(x[:, None]), y[:], "k-", lw=0.5)
            ax.plot(np.log10(x[~mask_, None]), y[~mask_], "*", c="k")
            ax.plot(np.log10(x[mask_, None]), y[mask_], "*", c="r")

        # RANSAC
        rs_estimator = lm.RANSACRegressor()
        rs_estimator.fit(X_[:, None], Y_)
        y_rs = rs_estimator.estimator_.predict(np.log10(x)[:, None])
        ax.plot(np.log10(x), y_rs, "-", label="RANSAC")

        # Theil Sen regression
        ts_model = lm.TheilSenRegressor(n_jobs=16)
        ts_model.fit(X_[:, None], Y_)
        y_ts = ts_model.predict(np.log10(x)[:, None])
        ax.plot(np.log10(x), y_ts, "--", label="Theil-Sen")

        # Huber regression
        hb_model = lm.HuberRegressor()
        hb_model.fit(X_[:, None], Y_)
        y_hb = hb_model.predict(np.log10(x)[:, None])
        ax.plot(np.log10(x), y_hb, "-.", label="Huber")

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
        ax.set_ylim([-10, 1])

    file_name = Path(f"{pwd}/png/outlier_detection.png")
    plib.save_fig(fig, file_name)


def main():
    cluster_list = sorted(src.glob('*.pq'))
    cluster = cluster_list[-1]

    df = get_pq(cluster)
    plot_outliers(df)


if __name__ == "__main__":
    main()
