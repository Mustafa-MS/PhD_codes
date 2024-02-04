import os
import gc
import time
import random
import numpy as np
import pandas as pd
import scipy.ndimage
import tensorflow as tf
import SimpleITK as sitk
import tensorflow_addons as tfa

base_path = '/home/mustafa/project/LUNA16/'  # Path to the LUNA16 folder
output_csv_path = '/home/mustafa/project/sliding_results001.csv'

# Load the truth table and filter for seriesuids with state = 1
truth_table = pd.read_csv('/home/mustafa/project/LUNA16/candidates_V2.csv')
seriesuids_with_nodules = truth_table[truth_table['class'] == 1]['seriesuid'].unique()
print(f"Total seriesuids with nodules: {len(seriesuids_with_nodules)}")

# Randomly sample 5% of the seriesuids
sample_size = int(len(seriesuids_with_nodules) * 0.002)  # 5% of the total
sampled_seriesuids = random.sample(list(seriesuids_with_nodules), sample_size)
print(f"Total seriesuids sampled for processing: {len(sampled_seriesuids)}")

all_results = []

def read_mhd_file(filepath):
    """Read and load CT image"""
    # Read file
    scan = sitk.ReadImage(filepath)
    # get the image to an array
    scan_array = sitk.GetArrayFromImage(scan)
    # Read the origin of the image
    origin = np.array(list(scan.GetOrigin()))[::-1]  # get [z, y, x] origin
    # Read spacing of the image
    old_spacing = np.array(list(scan.GetSpacing()))[::-1]  # get [z, y, x] spacing
    return scan_array, origin, old_spacing


def normalize(volume):
    """Normalize the CT image"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

def resample(image, old_spacing, new_spacing=[1, 1, 1]):
    """Resample to uniform the spacing"""
    resize_factor = old_spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = old_spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    return image, new_spacing

def load_ct_image(path):
    """Read the CT image, normalize, resample, and crop and save the nodules"""
    # Read scan
    volume, origin, old_spacing = read_mhd_file(path)
    # Normalize
    volume = normalize(volume)
    # Resample
    image, new_spacing = resample(volume, old_spacing)

    return image, new_spacing, origin

class BalancedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='balanced_accuracy', **kwargs):
        super(BalancedAccuracy, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.greater_equal(y_pred, 0.5)  # assuming your model outputs probabilities and threshold is 0.5

        self.true_positives.assign_add(
            tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True)), tf.float32)))
        self.true_negatives.assign_add(
            tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, False)), tf.float32)))
        self.false_positives.assign_add(
            tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True)), tf.float32)))
        self.false_negatives.assign_add(
            tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False)), tf.float32)))

    def result(self):
        sensitivity = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        specificity = self.true_negatives / (self.true_negatives + self.false_positives + tf.keras.backend.epsilon())
        return (sensitivity + specificity) / 2

    def reset_state(self):
        self.true_positives.assign(0.)
        self.true_negatives.assign(0.)
        self.false_positives.assign(0.)
        self.false_negatives.assign(0.)

model = tf.keras.models.load_model('/home/mustafa/project/resnet.keras',custom_objects={"BalancedAccuracy": BalancedAccuracy(), "F1Score": tfa.metrics.F1Score(num_classes=1, threshold=0.5)})

def predict_with_scanning_window(ct_image, seriesuid, spacing, origin, window_size= 31, stride=10):
    """
    Applies a scanning window over a CT image and uses a CNN model to make predictions.
    Also includes metadata with each prediction.

    Args:
    ct_image (numpy.ndarray): The CT image as a numpy array.
    window_size (tuple): The size of the scanning window (x, y, z).
    stride (int): The step size for moving the window.
    seriesuid (str): The unique identifier of the CT scan.
    origin (array-like): The origin coordinates of the CT scan.
    spacing (array-like): The spacing values of the CT scan.

    Yields:
    dict: Information about each prediction, including the predicted state, seriesuid, coords, and ct_metadata.
    """
    print("ct image shape: ", ct_image.shape)
    z_max, y_max, x_max = ct_image.shape
    time.sleep(5)
    for x in range(0, x_max - window_size + 1, stride):
        for y in range(0, y_max - window_size + 1, stride):
            for z in range(0, z_max - window_size + 1, stride):
                sub_volume = ct_image[ z:z + window_size, y:y + window_size, x:x + window_size]
                sub_volume = np.expand_dims(sub_volume, axis=-1)
                sub_volume = np.expand_dims(sub_volume, axis=0)
                x_center = x + (window_size // 2)
                y_center = y + (window_size // 2)
                z_center = z + (window_size // 2)
                # Make a prediction
                prediction = model.predict(sub_volume)
                # Prepare metadata for this prediction
                prediction_info = {
                    'predicted_state': prediction,
                    'seriesuid': seriesuid,
                    'coords': (z_center, y_center, x_center),
                    'ct_metadata': {'origin': origin, 'spacing': spacing}
                }
                print("prediction_info", prediction_info)
                yield prediction_info

def process_single_ct_scan(seriesuid, base_path, truth_table, distance_threshold=20):
    results = []
    found = False
    counter = 0
    dstnce_lst = []
    vox_lst = []
    for i in range(10):  # Check each subset
        subset_folder = f'subset{i}'
        ct_image_path = os.path.join(base_path, subset_folder, seriesuid + '.mhd')

        if os.path.exists(ct_image_path):
            found = True
            print(f"Processing {seriesuid} in {subset_folder}")
            ct_image, spacing, origin = load_ct_image(ct_image_path)
            predictions = predict_with_scanning_window(ct_image, seriesuid, spacing, origin)
            # Find matching nodules in ground truth
            matching_nodules = truth_table[(truth_table['seriesuid'] == seriesuid) & (truth_table['class'] == 1)]
            print("matching_nodules", len(matching_nodules))
            time.sleep(5)
            for _, matching_nodule in matching_nodules.iterrows():
                nod_vox_coords = (matching_nodule['coordZ'], matching_nodule['coordY'], matching_nodule['coordX'])
                nod_vox_coords = np.rint((nod_vox_coords - origin) / spacing)
                nod_vox_coords = np.array(nod_vox_coords, dtype=int)
                vox_lst.append(nod_vox_coords)

            # Process predictions and write to CSV incrementally
            for prediction in predictions:

                if prediction['predicted_state'] >= 0.6:  # If the model predicts a nodule
                    counter = counter +1
                    sub_volume_coords = prediction['coords']  # Coordinates of the sub-volume in image space
                    is_nodule_within_threshold = False

                    for _, nodule in matching_nodules.iterrows():
                        nodule_coords = (nodule['coordZ'], nodule['coordY'], nodule['coordX'])
                        nodule_coords = np.rint((nodule_coords - origin) / spacing)
                        nodule_coords = np.array(nodule_coords, dtype=int)
                        # this works but np.ling.norm is better
                        #distance = np.subtract(np.array(sub_volume_coords), np.array(nodule_coords))
                        distance = np.linalg.norm(np.array(sub_volume_coords) - np.array(nodule_coords))
                        #distance = np.absolute(distance)
                        dstnce_lst.append(distance)
                        # Check if the distance is within the threshold
                        #if max(distance) <= 15:
                        if distance <= 26:
                            is_nodule_within_threshold = True
                            break

                    state = prediction['predicted_state']
                    results.append({
                        'seriesuid': seriesuid,
                        'state': state,
                        'within threshold': is_nodule_within_threshold,
                        'z': sub_volume_coords[0],
                        'y': sub_volume_coords[1],
                        'x': sub_volume_coords[2]
                    })
            # Release memory
            del ct_image, predictions
            gc.collect()  # Explicit garbage collection
    if not found:
        print(f"CT image for seriesuid {seriesuid} not found in any subset.")

    return results, counter, matching_nodules, vox_lst, dstnce_lst


all_results = []
for index, seriesuid in enumerate(sampled_seriesuids):
    print(f"Processing {index + 1}/{len(sampled_seriesuids)}: {seriesuid}")
    result, counter, match_nodules, vox_coords, dist = process_single_ct_scan(seriesuid, base_path, truth_table, distance_threshold=15)
    all_results.extend(result)
    gc.collect()  # Explicit garbage collection after each seriesuid is processed
results_df = pd.DataFrame(all_results)
