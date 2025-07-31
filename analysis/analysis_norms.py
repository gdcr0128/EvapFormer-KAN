import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import pandas as pd

def load_npz_files(folder_path, key_name):
    """Load all .npz files from the specified folder and extract data using the provided key."""
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npz')])
    loaded_files = []
    for file in files:
        data = np.load(os.path.join(folder_path, file), allow_pickle=True)
        loaded_files.append(data[key_name])
    return loaded_files

def get_norms_values(norms_files):
    """Stack all norms arrays from the files into a single 3D array."""
    all_norms_values = np.stack(norms_files, axis=0)
    return all_norms_values

def visualize_and_save(data, title, save_path):
    """Visualize the mean of norms across samples and save the heatmap."""
    mean_norms_values = np.mean(data, axis=0)
    plt.figure(figsize=(8, 8))
    sns.heatmap(mean_norms_values, cmap='coolwarm', annot=False, cbar=True)
    plt.title(title)
    plt.xlabel("Sequence Position")
    plt.ylabel("Batch Index")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def print_feature_ranking(norms, labels):
    """Print the ranking of features based on their average norms."""
    labels = list(labels)  # Ensure labels is a list for indexing

    # Calculate the mean norms across all batches and features if norms is 3D
    if norms.ndim == 3:
        mean_norms = norms.mean(axis=0).mean(axis=0)
    else:
        mean_norms = norms.mean(axis=0)

    # Sort indices by descending mean norms
    sorted_indices = np.argsort(-mean_norms)

    print("基于范数的特征排名：")
    for idx in sorted_indices:
        print(f"{labels[min(idx, len(labels) - 1)]}: {mean_norms[idx]:.4f}")

def clear_folder(folder_path):
    """Delete all files and folders in the specified directory."""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def load_labels_from_csv_first_row(csv_path):
    """Load labels from the first row of the CSV file, excluding the first column."""
    df = pd.read_csv(csv_path, nrows=0)
    labels = df.columns.tolist()[1:]
    return labels

def main():
    norms_folder = '../norms'  # Path to the norms folder
    norms_key = 'norms'  # Key to extract data from .npz files
    csv_path = '../data/ERA5.csv'  # Path to the CSV dataset

    norms_files = load_npz_files(norms_folder, norms_key)
    labels = load_labels_from_csv_first_row(csv_path)
    norms_values = get_norms_values(norms_files)

    visualize_and_save(norms_values, "Average Transformed Vector Norms", "transformed_norms_visualization.png")
    print_feature_ranking(norms_values, labels)
    clear_folder(norms_folder)

if __name__ == "__main__":
    main()
