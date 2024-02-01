import pandas as pd
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import scipy
import tensorflow as tf
import csv
import os
import tensorflow_addons as tfa
# Load the truth table and filter for seriesuids with nodules
truth_table = pd.read_csv('/home/mustafa/project/LUNA16/candidates_V2.csv')
nodules = truth_table[truth_table['class'] == 1]
random_nodule = nodules.sample().iloc[0]  # Take a random nodule
seriesuid = random_nodule['seriesuid']

def read_mhd_file(filepath):
    """Read and load CT image"""
    # Read file
    #print("filepath for IMG = ", filepath)
    scan = sitk.ReadImage(filepath)
    # get the image to an array
    scan_array = sitk.GetArrayFromImage(scan)
    #[0, :, :, :]
    # Read the origin of the image
    origin = np.array(list(reversed(scan.GetOrigin())))  # get [z, y, x] origin
    # Delete the first element from origin
    #origin = np.delete(origin, 0)
    # Read spacing of the image
    old_spacing = np.array(list(reversed(scan.GetSpacing())))  # get [z, y, x] spacing
    # Delete the first element from spacing
    #old_spacing = np.delete(old_spacing, 0)
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


# This is just for testing

def worldToVoxelCoord(worldCoord, origin, spacing):
    # There is no need for this function, the cropping will handle the voxel coords
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    #voxelCoord = worldCoord
    return voxelCoord

def process_scan(path):
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

# Function to load a CT scan
def load_ct_scan(seriesuid, base_path):
    for i in range(10):  # Iterate through each subset
        subset_folder = f'subset{i}'
        ct_image_path = os.path.join(base_path, subset_folder, seriesuid + '.mhd')
        if os.path.exists(ct_image_path):

            ct_image, new_spacing, origin =   process_scan(ct_image_path)

            return ct_image, new_spacing, origin

    raise FileNotFoundError(f"CT scan for seriesuid {seriesuid} not found in any subset.")


base_path = '/home/mustafa/project/LUNA16/'
ct_scan, spacing, origin = load_ct_scan(seriesuid, base_path)
# Load your trained model
model = tf.keras.models.load_model('/home/mustafa/project/resnet.keras',custom_objects={"BalancedAccuracy": BalancedAccuracy(), "F1Score": tfa.metrics.F1Score(num_classes=1, threshold=0.5)})

def predict_nodule(ct_scan, model, world_coord, origin, spacing, window_size=(31,31,31)):
    voxel_coord = worldToVoxelCoord(world_coord, origin, spacing)
    #x, y, z = voxel_coord
    z, y, x = voxel_coord
    ct_shape = ct_scan.shape

    # Calculate start and end points for sub-volume extraction
    #start_x = max(int(x - window_size[0] // 2), 0)
    #end_x = min(start_x + window_size[0], ct_shape[0])
    #start_y = max(int(y - window_size[1] // 2), 0)
    #end_y = min(start_y + window_size[1], ct_shape[1])
    #start_z = max(int(z - window_size[2] // 2), 0)
    #end_z = min(start_z + window_size[2], ct_shape[2])
    # Calculate bounds for sub-volume extraction
    half_window = [w // 2 for w in window_size]
    start = [max(int(coord - half), 0) for coord, half in zip((x, y, z), half_window)]
    end = [min(int(coord + half), dim) for coord, half, dim in zip((x, y, z), half_window, ct_shape)]
    print("Voxel Coordinates:", x, y, z)
    print("CT Scan Shape:", ct_shape)
    print("Start Indices:", start)
    print("End Indices:", end)

    sub_volume = ct_scan[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    #sub_volume = ct_scan[start_x:end_x, start_y:end_y, start_z:end_z]
    # Before padding
    print("sub_volume shape:", sub_volume.shape)
    print("Expected window size:", window_size)
    padding = [(0, max(window_size[i] - sub_volume.shape[i], 0)) for i in range(3)]
    print("Padding:", padding)
    # Zero-pad the sub-volume if it's smaller than expected size
    if sub_volume.shape != window_size:
        padding = [(0, window_size[i] - sub_volume.shape[i]) for i in range(3)]
        sub_volume = np.pad(sub_volume, padding, mode='constant', constant_values=0)

    # Reshape sub_volume to include the channel dimension and normalize
    sub_volume = sub_volume.astype(np.float32)
    sub_volume = (sub_volume - np.min(sub_volume)) / (np.max(sub_volume) - np.min(sub_volume))
    sub_volume_reshaped = sub_volume.reshape((*window_size, 1))
    print("img_crop shape= ", sub_volume_reshaped.shape)
    imgg = np.expand_dims(sub_volume_reshaped, axis=0)
    print("imgg shape= ", imgg.shape)
    prediction = model.predict(imgg)
    predicted_state = np.argmax(prediction, axis=1)
    return predicted_state
def crop_nodule(image, new_spacing, window_size, origin, world_coord):
    """Cropping the nodule in 3D cube"""
    # Attention: Z, Y, X
    #nodule_center = np.array([patient.coordZ, patient.coordY, patient.coordX])
    nodule_center = world_coord
    # You can use the following line to convert from world to voxel coords.
    # voxelCoord = worldToVoxelCoord(nodule_center, origin, new_spacing)
    # The following line will do the same math so no need for converting from world to voxel for the centre coords
    v_center = np.rint((nodule_center - origin) / new_spacing)
    v_center = np.array(v_center, dtype=int)
    # This is to creat the cube Z Y X
    zyx_1 = v_center - window_size  # Attention: Z, Y, X
    zyx_2 = v_center + window_size + 1
    # This will give you a [19, 19, 19] volume
    img_crop = image[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]]
    img_crop = np.expand_dims(img_crop, axis=-1)
    print ("img_crop shape= ", img_crop.shape)
    img = np.expand_dims(img_crop, axis=0)
    prediction = model.predict(img)
    predicted_state = np.argmax(prediction, axis=1)
    return img_crop, prediction
results = []
window_size = 15

def plot_nodule(nodule_crop):
    # Learned from ArnavJain
    # https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing
    f, plots = plt.subplots(int(nodule_crop.shape[0] / 4) + 1, 4, figsize=(10, 10))

    for z_ in range(nodule_crop.shape[0]):
        plots[int(z_ / 4), z_ % 4].imshow(nodule_crop[z_, :, :], cmap='gray')

    # The last subplot has no image because there are only 19 images.
    plt.show()


for _, row in nodules[nodules['seriesuid'] == seriesuid].iterrows():

    world_coord = np.array([row['coordZ'], row['coordY'], row['coordX']])
    img, predicted_state = crop_nodule(ct_scan, spacing, window_size, origin, world_coord)
    plot_nodule(img)
    #predicted_state = predict_nodule(ct_scan, model, world_coord, origin, spacing)
    results.append({
        'seriesuid': seriesuid,
        'truth_coord': world_coord,
        'predicted_state': predicted_state[0],
        'truth_class': row['class']
    })
    for result in results:
        print("Truth Coordinates:", result['truth_coord'], "Predicted State:", result['predicted_state'], "Truth Class:",
          result['truth_class'])





'''
# Save results to CSV
with open('/home/mustafa/project/metrics/debug_results.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    for result in results:
        writer.writerow(result)
        print("Truth Coordinates:", result['truth_coord'], "Predicted State:", result['predicted_state'], "Truth Class:", result['truth_class'])
        plot_sub_volume(ct_scan, result['truth_coord'])
'''

