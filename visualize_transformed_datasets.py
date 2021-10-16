import matplotlib.pyplot as plt
import numpy as np


def visualize_ver_transf(dataset, version, n_series, transf):
    with open(f'{dataset}_version_{version}_10samples.npy', 'rb') as f_new, open(f'{dataset}_original.npy', 'rb') as f:
        y = np.load(f)
        y_new = np.load(f_new)
    
    fig, ax = plt.subplots(2, int(np.floor(n_series/2)), sharex='all')
    ax = ax.ravel()

    for i in range(y_new.shape[0]):
        for j in range(n_series):
            if i % 9 == 0 and not i == 0:
                ax[j].plot(y[i, :, j], label='original', color='darkorange')
                ax[j].set_title(f'{transf[j]}, s={j}')
            ax[j].plot(y_new[i, :, j], color='black', alpha=0.3)

    plt.legend()
    fig.suptitle(f'10 samples of {n_series} of the transformed series')
    plt.show()


def visualize_series_transf(dataset, n_series, n_versions, transf):
    with open(f'{dataset}_original.npy', 'rb') as f:
        y = np.load(f)

    y_ver = []
    for version in range(1, n_versions+1):
        with open(f'{dataset}_version_{version}_10samples.npy', 'rb') as f_new:
            y_new = np.load(f_new)
            y_ver.append(y_new)
    y_ver = np.array(y_ver)

    _, ax = plt.subplots(2, int(np.floor(n_versions/2)), sharex=True)
    ax = ax.ravel()

    colors = plt.cm.Blues_r(np.linspace(0, 0.65, n_versions))[::-1]

    for i in range(y_ver.shape[0]):
        for j in range(n_series):
            if (i+1) % 6 == 0:
                ax[j].plot(y[i, :, j], label='original', color='darkorange')
            ax[j].plot(y_ver[i, 0, :, j], label=f'version {i}', color=colors[i])
        ax[i].set_title(f'Series {i}')
    plt.legend()
    plt.show()


