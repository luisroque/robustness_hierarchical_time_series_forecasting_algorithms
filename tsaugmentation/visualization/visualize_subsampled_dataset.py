import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_series(dataset_missing, dataset_original, num_series=4):
    train_data_missing = dataset_missing['train']
    train_data_original = dataset_original['train']

    x_values_missing = train_data_missing['x_values']
    x_values_original = train_data_original['x_values']

    data_matrix_missing = train_data_missing['data']
    data_matrix_original = train_data_original['data']

    group_names = train_data_missing['groups_names']
    group_idx = train_data_missing['groups_idx']

    selected_series = np.random.choice(train_data_missing['n_series_idx'], num_series)

    n_rows = num_series
    n_cols = 1
    # n_rows = int(np.ceil(np.sqrt(num_series)))
    # n_cols = int(np.ceil(num_series / n_rows))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(14, 14))
    axs = axs.flatten()

    df_missing = pd.DataFrame(data_matrix_missing)
    df_original = pd.DataFrame(data_matrix_original)

    for ax, idx in zip(axs, selected_series):
        title_parts = []

        # Loop through each group name to dynamically generate the title and other features
        for group, names in group_names.items():
            group_value = names[group_idx[group][idx]]
            title_parts.append(f"{group.capitalize()}: {group_value}")

        ax.set_title(", ".join(title_parts), fontsize=12)

        y_values_missing = df_missing.iloc[:, idx]
        y_values_original = df_original.iloc[:, idx]

        # Interpolate the missing data to the same size as the original data
        y_values_interpolated = np.interp(x_values_original, x_values_missing, y_values_missing)

        ax.scatter(x_values_original, y_values_interpolated, marker='*', label="Interpolated", zorder=2)
        ax.plot(x_values_original, y_values_interpolated, label=None, zorder=1)
        ax.plot(x_values_original, y_values_original, label="Original", linestyle='--', zorder=0)

        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.legend(fontsize=8)

    # Remove any extra subplots
    for i in range(len(selected_series), n_rows * n_cols):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.show()

