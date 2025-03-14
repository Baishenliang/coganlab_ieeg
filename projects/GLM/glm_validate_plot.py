import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

def check_multicollinearity(X_i):
    """
    Computes the Variance Inflation Factor (VIF) for each feature in the given matrix
    to detect multicollinearity. Also, plots the correlation matrix as a heatmap.

    Parameters:
    X_i : numpy.ndarray
        Feature matrix where each column represents a feature.

    Returns:
    pd.DataFrame
        A DataFrame showing the VIF values for each feature.
    """

    # Ensure the input is a NumPy array
    if not isinstance(X_i, np.ndarray):
        raise ValueError("Input X_i must be a NumPy array")

    # Compute VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Feature"] = [f"X{i}" for i in range(X_i.shape[1])]
    vif_data["VIF"] = [variance_inflation_factor(X_i, i) for i in range(X_i.shape[1])]

    # Plot correlation matrix as a heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = np.corrcoef(X_i, rowvar=False)  # Compute correlation matrix
    sns.heatmap(correlation_matrix, annot=True, fmt=".1f", cmap="coolwarm", xticklabels=vif_data["Feature"],
                yticklabels=vif_data["Feature"], annot_kws={"size": 5})
    plt.title("Feature Correlation Matrix")
    plt.show()

    return vif_data

# Plot GLM
def plot_patient(data_in, title, colbar_lab,times):
    plt.figure(figsize=(10, 5))
    plt.imshow(data_in, aspect='auto', cmap='gray_r', interpolation='nearest')  # , vmin=-0.5e-5, vmax=0.5e-5)
    plt.colorbar(label=colbar_lab)
    plt.xlabel("Time points")
    plt.ylabel("Channels")
    plt.title(title)

    xticks = np.arange(0, len(times), 20)
    plt.xticks(ticks=xticks, labels=np.round(times[xticks], 2))
    plt.xlabel("Time (s)")
    #
    # yticks = np.arange(0, len(chs), 5)
    # plt.yticks(ticks=yticks, labels=[chs[i] for i in yticks])
    plt.ylabel("Channels")

    zero_time_index = np.argmin(np.abs(times - 0))
    plt.axvline(x=zero_time_index, color='red', linestyle='--', linewidth=1.5, label="Time = 0")

    plt.show()