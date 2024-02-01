import numpy as np
import matplotlib.pyplot as plt
import os
import random


def load_random_nodule_image(folder_path):
    """Load a random .npy file from the specified folder."""
    files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    if not files:
        raise FileNotFoundError("No .npy files found in the specified folder.")

    random_file = random.choice(files)
    file_path = os.path.join(folder_path, random_file)
    nodule_image = np.load(file_path)
    return nodule_image


def plot_nodule_slices(nodule_image):
    """Plot all slices of the nodule image in a grid."""
    num_slices = nodule_image.shape[0]
    cols = int(np.ceil(np.sqrt(num_slices)))
    rows = int(np.ceil(num_slices / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.ravel()

    for i in range(rows * cols):
        ax = axes[i]
        if i < num_slices:
            ax.imshow(nodule_image[i, :, :], cmap='gray')
            ax.set_title(f'Slice {i + 1}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# Usage example
folder_path = '/home/mustafa/project/LUNA16/cropped_nodules/'  # Replace with your folder path
pwd = os.getcwd()
print("pwd= ", pwd)
nodule_image = load_random_nodule_image(folder_path)
plot_nodule_slices(nodule_image)
