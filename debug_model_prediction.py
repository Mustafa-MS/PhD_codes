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
    scan = sitk.ReadImage(filepath)
    # get the image to an array
    scan_array = sitk.GetArrayFromImage(scan)
    # Read the origin of the image
    origin = np.array(list(reversed(scan.GetOrigin())))  # get [z, y, x] origin
    # Read spacing of the image
    old_spacing = np.array(list(reversed(scan.GetSpacing())))  # get [z, y, x] spacing
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

def crop_predict_nodule(image, new_spacing, window_size, origin, world_coord):
    """Cropping the nodule in 3D cube"""
    # Attention: Z, Y, X
    #nodule_center = np.array([patient.coordZ, patient.coordY, patient.coordX])
    nodule_center = world_coord
    print("nodule_center ", nodule_center)
    # The following line will do the same math so no need for converting from world to voxel for the centre coords
    v_center = np.rint((nodule_center - origin) / new_spacing)
    v_center = np.array(v_center, dtype=int)
    print("v_center ", v_center)
    # This is to creat the cube Z Y X
    zyx_1 = v_center - window_size  # Attention: Z, Y, X
    print("zyx_1 ", zyx_1)
    zyx_2 = v_center + window_size + 1
    print("zyx_2 ", zyx_2)
    # This will give you a [31, 31, 31] volume
    img_crop = image[zyx_1[0]:zyx_2[0], zyx_1[1]:zyx_2[1], zyx_1[2]:zyx_2[2]]
    img_crop = np.expand_dims(img_crop, axis=-1)
    img = np.expand_dims(img_crop, axis=0)
    prediction = model.predict(img)
    #predicted_state = np.argmax(prediction, axis=1)
    return img_crop, prediction
results = []
window_size = 15

def plot_nodule(nodule_crop):
    # Learned from ArnavJain
    # https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing
    f, plots = plt.subplots(int(nodule_crop.shape[0] / 4) + 1, 4, figsize=(10, 10))

    for z_ in range(nodule_crop.shape[0]):
        plots[int(z_ / 4), z_ % 4].imshow(nodule_crop[z_, :, :], cmap='gray')

    plt.show()


for _, row in nodules[nodules['seriesuid'] == seriesuid].iterrows():

    world_coord = np.array([row['coordZ'], row['coordY'], row['coordX']])
    img, predicted_state = crop_predict_nodule(ct_scan, spacing, window_size, origin, world_coord)
    plot_nodule(img)
    results.append({
        'seriesuid': seriesuid,
        'truth_coord': world_coord,
        'predicted_state': predicted_state[0],
        'truth_class': row['class']
    })
    for result in results:
        print("Truth Coordinates:", result['truth_coord'], "Predicted State:", result['predicted_state'], "Truth Class:",
          result['truth_class'])




# Save results to CSV
# with open('/home/mustafa/project/metrics/debug_results.csv', 'w', newline='') as file:
#     writer = csv.DictWriter(file, fieldnames=results[0].keys())
#     writer.writeheader()
#     for result in results:
#         writer.writerow(result)
#         print("Truth Coordinates:", result['truth_coord'], "Predicted State:", result['predicted_state'], "Truth Class:", result['truth_class'])
#         plot_sub_volume(ct_scan, result['truth_coord'])


