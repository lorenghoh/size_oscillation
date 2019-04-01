import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def init_plot():
    # TODO: user-defined sizes
    fig = plt.figure(figsize=(12, 4))

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

    return fig

def save_fig(fig, file_name):
    fig.tight_layout(pad=0.5)
    print('\t Writing figure to {}...'.format(file_name))
    fig.savefig(
        file_name,
        bbox_inches='tight',
        dpi=180,
        facecolor='w', 
        transparent=True
    )