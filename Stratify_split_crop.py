import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import SimpleITK as sitk
import gc


# Load the truth table
truth_df = pd.read_csv('/home/mustafa/project/dataset/candidates_V2.csv')
data_dir = '/home/mustafa/project/dataset'
preprocessed_base_dir = '/home/mustafa/project/preprocessed_scans'
cropped_base_dir = '/home/mustafa/project/cropped_scans'
os.makedirs(os.path.join(preprocessed_base_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(preprocessed_base_dir, 'val'), exist_ok=True)

print(truth_df.head())
print(truth_df.columns)

# Group candidates by scan and determine if the scan contains any nodules
scan_labels = truth_df.groupby('seriesuid')['class'].max().reset_index()
scan_labels.rename(columns={'class': 'has_nodule'}, inplace=True)
print(scan_labels.head())


# Extract the seriesuids and labels
scan_ids = scan_labels['seriesuid'].values
scan_nodule_labels = scan_labels['has_nodule'].values

# First, split into training and temp (for validation and test)
train_scans, temp_scans, train_labels, temp_labels = train_test_split(
  scan_ids, scan_nodule_labels, test_size=0.2, random_state=42, stratify=scan_nodule_labels)

# Now split temp into validation and test sets
val_scans, test_scans, val_labels, test_labels = train_test_split(
  temp_scans, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)

print(f'Number of training scans: {len(train_scans)}')
print(f'Number of validation scans: {len(val_scans)}')
print(f'Number of test scans: {len(test_scans)}')


def load_itk_image(scan_id):
  import os
  import SimpleITK as sitk

  # Path to the directory containing the subsets
  #data_dir = '/home/dataset'  # Adjust if your data directory is different

  # Flag to indicate if the scan was found
  found = False

  # Iterate over the subset directories to find the scan
  for subset_idx in range(10):
      subset_path = os.path.join(data_dir, f'subset{subset_idx}')
      mhd_path = os.path.join(subset_path, f'{scan_id}.mhd')
      if os.path.exists(mhd_path):
          found = True
          break  # Stop searching once the scan is found

  if not found:
      print(f'Scan {scan_id} not found in any subset.')
      return None

  # Read the scan using SimpleITK
  itk_image = sitk.ReadImage(mhd_path)

  return itk_image


# Function to calculate the proportion of scans with nodules
def calculate_nodule_proportion(scan_list, scan_labels_df):
  labels = scan_labels_df.loc[scan_labels_df['seriesuid'].isin(scan_list), 'has_nodule']
  proportion = labels.mean()
  total_scans = len(labels)
  nodule_scans = labels.sum()
  print(f'Total scans: {total_scans}, Scans with nodules: {nodule_scans}, Proportion: {proportion:.2f}')

print('Training set:')
calculate_nodule_proportion(train_scans, scan_labels)

print('Validation set:')
calculate_nodule_proportion(val_scans, scan_labels)

print('Test set:')
calculate_nodule_proportion(test_scans, scan_labels)

# Create a mapping from seriesuid to set
scan_to_set = {}
for scan_id in train_scans:
  scan_to_set[scan_id] = 'train'
for scan_id in val_scans:
  scan_to_set[scan_id] = 'val'
for scan_id in test_scans:
  scan_to_set[scan_id] = 'test'

# Map the 'seriesuid' to the corresponding set
truth_df['set'] = truth_df['seriesuid'].map(scan_to_set)

# Drop any candidates where the set is NaN (shouldn't happen)
truth_df = truth_df.dropna(subset=['set'])

print(truth_df['set'].value_counts())


def resample_scan(itk_image, new_spacing=[1.0, 1.0, 1.0]):
    # Get current spacing
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    # Compute new size
    new_size = [
        int(np.round(original_size[i] * (original_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]
    # Resample
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetInterpolator(sitk.sitkLinear)
    new_itk_image = resample.Execute(itk_image)
    return new_itk_image



def compute_mean_and_std_with_sampling(scan_ids, num_samples_per_scan=100000):
  samples = []

  for scan_id in scan_ids:
      print(f'Processing scan {scan_id} for mean and std computation')
      itk_image = load_itk_image(scan_id)
      if itk_image is None:
          continue

      image_array = sitk.GetArrayFromImage(itk_image).astype(np.float32)
      # Apply preprocessing steps like resampling or clipping if desired

      flat_array = image_array.flatten()
      # Sample a subset of voxels randomly
      if flat_array.size > num_samples_per_scan:
          indices = np.random.choice(flat_array.size, num_samples_per_scan, replace=False)
          scan_samples = flat_array[indices]
      else:
          scan_samples = flat_array  # Use all voxels if less than num_samples_per_scan

      samples.append(scan_samples)

      # Free memory
      del image_array
      del flat_array

  # Concatenate samples from all scans
  all_samples = np.concatenate(samples)

  # Compute mean and std from the samples
  voxel_mean = np.mean(all_samples)
  voxel_std = np.std(all_samples)

  return voxel_mean, voxel_std

voxel_mean, voxel_std = compute_mean_and_std_with_sampling(train_scans)
print(f'Sampled voxel mean: {voxel_mean}')
print(f'Sampled voxel std: {voxel_std}')

def clip_intensity(image_array, min_value=-1000, max_value=400):
    image_array = np.clip(image_array, min_value, max_value)
    return image_array



def preprocess_scan(itk_image, voxel_mean=None, voxel_std=None):
  # Resample to isotropic resolution
    itk_image = resample_scan(itk_image)
    image_array = sitk.GetArrayFromImage(itk_image)
  # Clip intensity values
    image_array = clip_intensity(image_array)
  # Normalize using mean and std from training set
    if voxel_mean is not None and voxel_std is not None:
        image_array = (image_array - voxel_mean) / voxel_std
    else:
      # If mean and std are not provided, skip normalization
        pass
  # Convert back to ITK image
    preprocessed_itk_image = sitk.GetImageFromArray(image_array)
    preprocessed_itk_image.SetSpacing(itk_image.GetSpacing())
    preprocessed_itk_image.SetOrigin(itk_image.GetOrigin())
    preprocessed_itk_image.SetDirection(itk_image.GetDirection())
    return preprocessed_itk_image





def preprocess_and_save_scans(scan_ids, set_name, voxel_mean=None, voxel_std=None):
  for scan_id in scan_ids:
      print(f'Preprocessing scan {scan_id} for set {set_name}')
      itk_image = load_itk_image(scan_id)
      if itk_image is None:
          continue
      preprocessed_itk_image = preprocess_scan(itk_image, voxel_mean, voxel_std)
      # Save preprocessed scan
      save_path = os.path.join(preprocessed_base_dir, set_name, f'{scan_id}.mhd')
      sitk.WriteImage(preprocessed_itk_image, save_path)


preprocess_and_save_scans(train_scans, 'train', voxel_mean, voxel_std)
preprocess_and_save_scans(val_scans, 'val', voxel_mean, voxel_std)

def crop_and_save_patches(truth_df, set_name):
    # Get candidates for the set
    candidates = truth_df[truth_df['set'] == set_name]
    # Group candidates by scan
    grouped = candidates.groupby('seriesuid')

    for scan_id, group in grouped:
        print(f'Processing scan {scan_id} for set {set_name}')
        mhd_path = os.path.join(preprocessed_base_dir, set_name, f'{scan_id}.mhd')
        if not os.path.exists(mhd_path):
            print(f'Preprocessed scan {scan_id} not found.')
            continue

        itk_image = sitk.ReadImage(mhd_path)
        image_array = sitk.GetArrayFromImage(itk_image)
        origin = np.array(itk_image.GetOrigin())  # x, y, z
        spacing = np.array(itk_image.GetSpacing())  # x, y, z

        # Adjust for axes order difference between SimpleITK and NumPy
        spacing = spacing[[2, 1, 0]]
        origin = origin[[2, 1, 0]]

        # Find the path to the scan file
        # Iterate through subsets to find the scan
        found = False
        for subset_idx in range(10):
            subset_path = os.path.join(data_dir, f'subset{subset_idx}')
            mhd_path = os.path.join(subset_path, f'{scan_id}.mhd')
            if os.path.exists(mhd_path):
                found = True
                break
        if not found:
            print(f'Scan {scan_id} not found in any subset.')
            continue

        # Read the scan using SimpleITK
        itk_image = sitk.ReadImage(mhd_path)
        image_array = sitk.GetArrayFromImage(itk_image)  # Get NumPy array (z, y, x)
        origin = np.array(itk_image.GetOrigin())  # x, y, z
        spacing = np.array(itk_image.GetSpacing())  # x, y, z spacing

        # Adjust for axes order difference between SimpleITK and NumPy
        #spacing = spacing[[2, 1, 0]]
        #origin = origin[[2, 1, 0]]

        # Process each candidate in the scan
        for idx, candidate in group.iterrows():
            # Get world coordinates
            world_coord = np.array([candidate['coordZ'], candidate['coordY'],
                                    candidate['coordX']])  # Adjust order to match image_array axes
            # Convert world coordinates to voxel indices
            voxel_coord = np.rint((world_coord - origin) / spacing).astype(int)
            # Extract the patch centered at voxel_coord
            patch_size = 31  # Assuming patch size is 31x31x31 voxels
            half_size = patch_size // 2
            z, y, x = voxel_coord
            # Define the bounding box
            z_min = max(z - half_size, 0)
            y_min = max(y - half_size, 0)
            x_min = max(x - half_size, 0)
            z_max = min(z + half_size + 1, image_array.shape[0])
            y_max = min(y + half_size + 1, image_array.shape[1])
            x_max = min(x + half_size + 1, image_array.shape[2])
            # Extract the patch
            patch = image_array[z_min:z_max, y_min:y_max, x_min:x_max]
            # Pad the patch if necessary
            pad_z = (max(0, half_size - z), max(0, z + half_size + 1 - image_array.shape[0]))
            pad_y = (max(0, half_size - y), max(0, y + half_size + 1 - image_array.shape[1]))
            pad_x = (max(0, half_size - x), max(0, x + half_size + 1 - image_array.shape[2]))
            patch = np.pad(patch, (pad_z, pad_y, pad_x), mode='constant', constant_values=0)
            # Ensure the patch is the correct size
            assert patch.shape == (patch_size, patch_size, patch_size), f'Patch shape mismatch: {patch.shape}'
            # Save the patch
            # File naming: nodule_{idx}.npy
            label = candidate['class']
            save_dir = os.path.join(cropped_base_dir, set_name)
            os.makedirs(save_dir, exist_ok=True)
            filename = f'nodule_{idx}.npy'
            save_path = os.path.join(save_dir, filename)
            np.save(save_path, patch.astype(np.int16))  # Save as int16 to save space
            # Optionally, save the label in a CSV file
            with open(os.path.join(save_dir, 'labels.csv'), 'a') as f:
                f.write(f'{filename},{label}\n')


# Remove existing labels.csv files

for set_name in ['train', 'val']:
  labels_file = os.path.join(cropped_base_dir, set_name, 'labels.csv')
  if os.path.exists(labels_file):
      os.remove(labels_file)



crop_and_save_patches(truth_df, 'train')
crop_and_save_patches(truth_df, 'val')