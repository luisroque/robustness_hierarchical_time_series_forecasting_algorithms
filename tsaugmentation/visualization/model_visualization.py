import matplotlib.pyplot as plt
from keras.callbacks import History
import numpy as np


def plot_loss(history: History, first_index: int, dataset_name: str) -> None:
    """
    Plot total loss, reconstruction loss and kl_loss per epoch

    :param history: recorded loss
    :param first_index: first index of the loss arrays to plot to avoid hard to read plots
    :param dataset_name: name of the dataset to plot and store
    """

    _, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(history.history["loss"][first_index:])
    ax.plot(history.history["reconstruction_loss"][first_index:])
    ax.plot(history.history["kl_loss"][first_index:])
    ax.set_title("model loss")
    ax.set_ylabel("loss")
    ax.set_xlabel("epoch")
    plt.legend(["total_loss", "reconstruction_loss", "kl_loss"], loc="upper left")
    plt.savefig(
        f"./plots/vae_loss_{dataset_name}.pdf", format="pdf", bbox_inches="tight"
    )
    plt.show()


def plot_generated_vs_original(
    dec_pred_hat: np.ndarray,
    X_train_raw: np.ndarray,
    param_vae: float,
    dataset_name: str,
    n_series: int = 8,
    directory: str = ".",
) -> None:
    """
    Plot generated series and the original series and store as pdf

    Args:
        dec_pred_hat: predictions
        X_train_raw: original series
        param_vae: the parameter used in the VAE sampling
        dataset_name: name of the generated dataset
        n_series: number of series to plot
        directory: local directory to store the file
    """
    # n_series needs to be even
    if not n_series % 2 == 0:
        n_series -= 1
    _, ax = plt.subplots(int(n_series // 2), int(n_series // 4), figsize=(18, 10))
    ax = ax.ravel()
    n_samples = X_train_raw.shape[0]
    for i in range(n_series):
        ax[i].plot(np.arange(n_samples), dec_pred_hat[:, i], label="new sample")
        ax[i].plot(np.arange(n_samples), X_train_raw[:, i], label="orig")
    plt.legend()
    plt.suptitle(
        f"VAE generated dataset vs original -> {dataset_name} with sigma={param_vae}",
        fontsize=14,
    )
    plt.savefig(
        f"{directory}/plots/vae_generated_vs_original_{dataset_name}_{param_vae}.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.show()
