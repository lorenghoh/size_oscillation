import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

plt.style.use(str(Path(__file__).parent / "custom.mplstyle"))


def init_plot(fs=(12, 4)):
    fig = plt.figure(figsize=fs)

    sns.set_context("notebook")
    sns.set_style(
        "ticks",
        {
            "axes.grid": True,
            "axes.linewidth": "1",
            "grid.color": "0.5",
            "grid.linestyle": ":",
            "legend.frameon": True,
        },
    )

    rc_fonts = {
        "font.family": "serif",
        "text.usetex": False,
        "text.latex.preamble":
            r"\usepackage{libertine} \usepackage[libertine]{newtxmath}",
    }

    mpl.rcParams.update(rc_fonts)

    return fig


def close_fig(fig):
    plt.close(fig)


def save_fig(fig, file_path, print_output=True, rename_fig=True):
    if rename_fig:
        file_name = file_path.with_suffix(".png").name
        png_dir = Path(__file__).parents[2] / "png"

    if print_output:
        print(f"\t Writing figure to {png_dir}/{file_name}...")

    fig.tight_layout(pad=0.5)
    fig.savefig(
        Path(file_path), dpi=180, facecolor="w", transparent=True,
    )
