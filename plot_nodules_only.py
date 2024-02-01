import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random


def load_image_with_state(folder_path, csv_path, state=1):
    """Load a random .npy image file with the specified state."""
    df = pd.read_csv(csv_path)
    # Filter for images with the specified state
    valid_images = df[df['state'] == state]['SN'].tolist()

    if not valid_images:
        raise ValueError("No images with the specified state found.")

    selected_image_name = random.choice(valid_images)
    image_path = os.path.join(folder_path, str(selected_image_name) + '.npy')
    print("img path = ", image_path)
    #if not os.path.exists(image_path):
    #    raise FileNotFoundError(f"File {selected_image_name}.npy not found in the folder.")

    return np.load(image_path), selected_image_name


def plot_image_slices(image, title="Image Slices"):
    """Plot all slices of a 3D image."""
    num_slices = image.shape[0]
    cols = int(np.sqrt(num_slices))
    rows = int(np.ceil(num_slices / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.ravel()

    for i in range(rows * cols):
        if i < num_slices:
            axes[i].imshow(image[i], cmap='gray')
            axes[i].set_title(f'Slice {i + 1}')
            axes[i].axis('off')
        else:
            axes[i].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# Example usage
folder_path = '/home/mustafa/project/LUNA16/cropped_nodules/'  # Replace with the path to your images folder
csv_path = '/home/mustafa/project/LUNA16/cropped_nodules.csv'  # Replace with the path to your CSV file

pwd = os.getcwd()
print("pwd= ", pwd)
# Load the truth table and filter for seriesuids with nodules
truth_table = pd.read_csv('/home/mustafa/project/LUNA16/cropped_nodules.csv')
nodules = truth_table[truth_table['state'] == 1]
random_nodule = nodules.sample().iloc[0]  # Take a random nodule
seriesuid = str(random_nodule['SN'])
image_path = os.path.join(folder_path, seriesuid + '.npy')
print("img path = ", image_path)
image = np.load(image_path)
print("image shape = ", image.shape)
print("loaded image")
# Load and plot
#image, image_name = load_image_with_state(folder_path, csv_path, state=1)
plot_image_slices(image, title=f"Slices of {seriesuid}")
