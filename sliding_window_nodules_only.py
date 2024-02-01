# important note, this code is consuming memory!!

import SimpleITK as sitk
import numpy as np
import pandas as pd
import scipy.ndimage
import tensorflow as tf
import glob
import os

# Load the pre-trained model
model = tf.keras.models.load_model('3d_image_classification.h5')  # Replace with the actual path

# Load the truth table and filter for seriesuids with state = 1
truth_table = pd.read_csv('/home/mustafa/project/LUNA16/candidates_V2.csv')
seriesuids_with_nodules = truth_table[truth_table['class'] == 1]['seriesuid'].unique()

# Function definitions (voxel_2_world, calculate_precision, load_ct_image, scanning_window, predict_with_scanning_window)
# ... (Include all the necessary functions here)

# Function to convert voxel coordinates to world coordinates
def voxel_2_world(voxel_coordinates, origin, spacing):
    stretched_voxel_coordinates = voxel_coordinates * spacing
    world_coordinates = stretched_voxel_coordinates + origin
    return world_coordinates


# Function to calculate precision
def calculate_precision(predictions, truth_table, distance_threshold=20):
    TP = 0
    FP = 0

    for prediction in predictions:
        if prediction['predicted_state'] == 1:  # If the model predicts a nodule
            seriesuid = prediction['seriesuid']
            sub_volume_coords = prediction['coords']  # Coordinates of the sub-volume in image space
            ct_metadata = prediction['ct_metadata']

            # Convert sub-volume coordinates to world coordinates
            world_coords = voxel_2_world(sub_volume_coords, ct_metadata['origin'], ct_metadata['spacing'])

            # Find matching nodules in ground truth
            matching_nodules = truth_table[(truth_table['seriesuid'] == seriesuid) & (truth_table['class'] == 1)]

            # Check for any nodule within the distance threshold
            is_TP = any(np.linalg.norm(np.array(world_coords) - np.array((nodule['coordX'], nodule['coordY'], nodule['coordZ']))) <= distance_threshold for _, nodule in matching_nodules.iterrows())

            if is_TP:
                TP += 1
            else:
                FP += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    return precision

def resample(image, old_spacing, new_spacing=[1, 1, 1]):
    resize_factor = old_spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = old_spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing



def load_ct_image(path_to_mhd_file):
    """
    Loads a CT image from a Meta Image (.mhd) file.

    Args:
    path_to_mhd_file (str): Path to the .mhd file

    Returns:
    numpy.ndarray: The CT image as a numpy array
    """
    # Read the image using SimpleITK
    sitk_image = sitk.ReadImage(path_to_mhd_file)

    # Convert the image to a numpy array
    ct_image = sitk.GetArrayFromImage(sitk_image)
    origin = np.array(sitk_image.GetOrigin())[::-1]  # get [z, y, x] origin
    old_spacing = np.array(sitk_image.GetSpacing())[::-1]  # get [z, y, x] spacing

    image, new_spacing = resample(ct_image, old_spacing)
    # Normalize the image (example: windowing or intensity normalization)
    ct_image = ct_image.astype(np.float32)
    ct_image = (ct_image - np.mean(ct_image)) / np.std(ct_image)

    return ct_image, origin, new_spacing


# Now ct_image is a numpy array representing the CT image

def scanning_window(ct_image, window_size=(40, 40, 40), stride=20):
    """
    Generator function to iterate over a CT image using a 3D scanning window.

    Args:
    ct_image (numpy.ndarray): The CT image as a numpy array.
    window_size (tuple): The size of the scanning window (x, y, z).
    stride (int): The step size for moving the window.

    Yields:
    numpy.ndarray: Sub-volume extracted by the scanning window.
    """
    x_max, y_max, z_max = ct_image.shape

    for x in range(0, x_max - window_size[0] + 1, stride):
        for y in range(0, y_max - window_size[1] + 1, stride):
            for z in range(0, z_max - window_size[2] + 1, stride):
                yield ct_image[x:x + window_size[0], y:y + window_size[1], z:z + window_size[2]]




'''
def predict_with_scanning_window(ct_image, model, window_size=(31, 31, 31), stride=20):
    """
    Applies a scanning window over a CT image and uses a CNN model to make predictions.

    Args:
    ct_image (numpy.ndarray): The CT image as a numpy array.
    model (tf.keras.Model): The pre-trained CNN model.
    window_size (tuple): The size of the scanning window (x, y, z).
    stride (int): The step size for moving the window.

    Yields:
    numpy.ndarray: The prediction for each sub-volume.
    """

    for sub_volume in scanning_window(ct_image, window_size, stride):
        # Reshape sub_volume for the model if necessary
        #print("sub volume ",sub_volume.shape)
        sub_volume_reshaped = np.expand_dims(sub_volume, axis=0)

        # Make a prediction
        prediction = model.predict(sub_volume_reshaped)
        yield prediction
'''


def predict_with_scanning_window(ct_image, model, seriesuid, origin, spacing, window_size=(31, 31, 31), stride=20):
    """
    Applies a scanning window over a CT image and uses a CNN model to make predictions.
    Also includes metadata with each prediction.

    Args:
    ct_image (numpy.ndarray): The CT image as a numpy array.
    model (tf.keras.Model): The pre-trained CNN model.
    window_size (tuple): The size of the scanning window (x, y, z).
    stride (int): The step size for moving the window.
    seriesuid (str): The unique identifier of the CT scan.
    origin (array-like): The origin coordinates of the CT scan.
    spacing (array-like): The spacing values of the CT scan.

    Yields:
    dict: Information about each prediction, including the predicted state, seriesuid, coords, and ct_metadata.
    """

    x_max, y_max, z_max = ct_image.shape

    for x in range(0, x_max - window_size[0] + 1, stride):
        for y in range(0, y_max - window_size[1] + 1, stride):
            for z in range(0, z_max - window_size[2] + 1, stride):
                sub_volume = ct_image[x:x + window_size[0], y:y + window_size[1], z:z + window_size[2]]
                sub_volume_reshaped = np.expand_dims(sub_volume, axis=0)

                # Make a prediction
                prediction = model.predict(sub_volume_reshaped)
                predicted_state = np.argmax(prediction, axis=1)

                # Prepare metadata for this prediction
                prediction_info = {
                    'predicted_state': predicted_state[0],
                    'seriesuid': seriesuid,
                    'coords': (x, y, z),
                    'ct_metadata': {'origin': origin, 'spacing': spacing}
                }

                yield prediction_info

def process_ct_scans_for_seriesuids(seriesuids, model, truth_table, output_csv_path, distance_threshold=20):
    results = []
    base_path = '/home/mustafa/project/LUNA16/'  # Path to the LUNA16 folder

    for seriesuid in seriesuids:
        found = False
        for i in range(10):  # Iterate through each subset folder
            subset_folder = f'subset{i}'
            ct_image_path = os.path.join(base_path, subset_folder, seriesuid + '.mhd')

            if os.path.exists(ct_image_path):
                found = True
                ct_image, origin, spacing = load_ct_image(ct_image_path)
                predictions = predict_with_scanning_window(ct_image, model, seriesuid, origin, spacing)

                for prediction in predictions:
                    seriesuid = prediction['seriesuid']
                    ct_metadata = prediction['ct_metadata']
                    state = None
                    world_coords = None

                    if prediction['predicted_state'] == 1:  # If the model predicts a nodule
                        sub_volume_coords = prediction['coords']  # Coordinates of the sub-volume in image space

                        # Convert sub-volume coordinates to world coordinates
                        world_coords = voxel_2_world(sub_volume_coords, ct_metadata['origin'], ct_metadata['spacing'])

                        # Find matching nodules in ground truth
                        matching_nodules = truth_table[
                            (truth_table['seriesuid'] == seriesuid) & (truth_table['class'] == 1)]

                        # Check for any nodule within the distance threshold
                        is_TP = any(np.linalg.norm(np.array(world_coords) - np.array(
                            (nodule['coordX'], nodule['coordY'], nodule['coordZ']))) <= distance_threshold for _, nodule in
                                    matching_nodules.iterrows())
                        state = '1' if is_TP else 'FP'

                    # Append result for this scan
                    results.append({'seriesuid': seriesuid, 'state': state, 'x': world_coords[0] if world_coords else None,
                                    'y': world_coords[1] if world_coords else None,
                                    'z': world_coords[2] if world_coords else None})

                break  # Exit the loop once the file is found

        if not found:
            print(f"CT image for seriesuid {seriesuid} not found in any subset.")

    # Write results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, index=False)

    # Calculate precision
    return calculate_precision(results, truth_table)

# Example usage
output_csv_path = '/path/to/output/results.csv'
precision = process_ct_scans_for_seriesuids(seriesuids_with_nodules, model, truth_table, output_csv_path)
print(f'Precision: {precision}')
