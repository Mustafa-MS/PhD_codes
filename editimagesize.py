import os
import numpy as np
import pandas as pd
# Load labels from CSV
labels_df = pd.read_csv('/home/mustafa/project/processed_dataset2/train/train_info.csv')

# Prepare file paths and labels
image_dir = '/home/mustafa/project/processed_dataset2/train/samples'  # directory where .npy files are stored
#file_paths = [os.path.join(image_dir, f'{i}.npy') for i in labels_df['file_name']]
file_paths = [os.path.join(image_dir, f'{i}') for i in labels_df['file_name']]
# Function to find images with incorrect shapes
def find_incorrect_shape_images(file_paths, target_shape=(31, 31, 31)):
  incorrect_images = []
  for file_path in file_paths:
      image = np.load(file_path)
      if image.shape != target_shape:
          incorrect_images.append((file_path, image.shape))
  print(f'Number of images with incorrect shape: {len(incorrect_images)}')
  return incorrect_images

# Padding function
def pad_image_to_shape(image, target_shape=(31, 31, 31)):
  current_shape = image.shape
  pad_width = []
  for i in range(len(target_shape)):
      total_padding = target_shape[i] - current_shape[i]
      if total_padding >= 0:
          if total_padding % 2 == 0:
              padding = (total_padding // 2, total_padding // 2)
          else:
              padding = (total_padding // 2, total_padding // 2 + 1)
      else:
          raise ValueError(f"Image is larger than target shape in dimension {i}")
      pad_width.append(padding)
  # Pad the image with zeros
  padded_image = np.pad(image, pad_width=pad_width, mode='constant', constant_values=0)
  return padded_image

# Preprocess and overwrite images
def preprocess_and_overwrite_images(incorrect_images):
  for file_path, original_shape in incorrect_images:
      # Load the image
      image = np.load(file_path)
      # Pad the image to the target shape
      padded_image = pad_image_to_shape(image, target_shape=(31, 31, 31))
      # Save the padded image, overwriting the old file
      np.save(file_path, padded_image)
      print(f'Processed and saved {file_path}')

# Assuming 'file_paths' is your list of all image file paths
incorrect_images = find_incorrect_shape_images(file_paths)

# Preprocess images
#preprocess_and_overwrite_images(incorrect_images)

# Verify all images are now of correct shape
#incorrect_images_after = find_incorrect_shape_images(file_paths)
#if len(incorrect_images_after) == 0:
#  print('All images are now of the correct shape.')
#else:
#  print(f'Some images still have incorrect shapes: {len(incorrect_images_after)}')