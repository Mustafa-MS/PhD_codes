import os
import numpy as np
import pandas as pd

# Load labels from CSV
labels_df = pd.read_csv('/home/mustafa/project/dataset/cropped_nodules.csv')

# Prepare file paths and labels
image_dir = '/home/mustafa/project/dataset/cropped_nodules/'  # directory where .npy files are stored
file_paths = [os.path.join(image_dir, f'{i}.npy') for i in labels_df['SN']]

def check_image_shapes(file_paths):
  incorrect_shapes = []
  for file_path in file_paths:
      image = np.load(file_path)
      if image.shape != (31, 31, 31):
          incorrect_shapes.append((file_path, image.shape))
  print(f'Number of images with incorrect shape: {len(incorrect_shapes)}')
  for file_path, shape in incorrect_shapes:
      print(f'File: {file_path}, Shape: {shape}')
  return incorrect_shapes

# Call the function with all file paths
incorrect_images = check_image_shapes(file_paths)