import matplotlib
import matplotlib.pyplot as plt

from pathlib import Path

plt.style.use(str(Path(__file__).parent / "custom.mplstyle"))


def init_plot(figsize=(12, 4)):
    fig = plt.figure(figsize=figsize)

    # TODO: Matplolib style configuration

    plt.rc("text", usetex=True)
    plt.rc("font", family="Serif")

    return fig


def save_fig(fig, file_path, print_output=True):
    file_name = file_path.with_suffix(".png").name
    png_dir = Path(__file__).parents[2] / "png"

    if print_output:
        print(f"\t Writing figure to {png_dir}/{file_name}...")

    fig.tight_layout(pad=0.5)
    fig.savefig(
        Path(png_dir / file_name), 
        bbox_inches="tight", 
        dpi=180, 
        facecolor="w", 
        transparent=True,
    )
