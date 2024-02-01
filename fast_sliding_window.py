import numpy as np
import SimpleITK as sitk
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import TruePositives, FalsePositives, TrueNegatives, FalseNegatives, Precision, Recall, BinaryAccuracy


def load_ct_scan(path):
    sitk_image = sitk.ReadImage(path)
    ct_scan = sitk.GetArrayFromImage(sitk_image)
    # Normalize ct_scan
    ct_scan = ct_scan.astype(np.float32)
    ct_scan = (ct_scan - np.min(ct_scan)) / (np.max(ct_scan) - np.min(ct_scan))
    return ct_scan

def load_mask(mask_path):
    return np.load(mask_path)['arr_0']

# Example of loading a CT scan and its mask
ct_scan = load_ct_scan('/home/mustafa/project/LUNA16/')
mask = load_mask('/path/to/mask_folder/ct_scan_mask.npy')

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Conv3DTranspose

def create_conv_fast_sliding_window_model(input_shape):
    inputs = Input(input_shape)
    x = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling3D((2, 2, 2))(x)
    # Add more layers as necessary...
    outputs = Conv3D(1, (1, 1, 1), activation='sigmoid', padding='same')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

model = create_conv_fast_sliding_window_model((128, 128, 128))  # Adjust the input shape based on your data


def data_generator(ct_scans_paths, masks_paths, batch_size):
    while True:
        batch_paths = np.random.choice(a=len(ct_scans_paths), size=batch_size)
        batch_ct_scans = []
        batch_masks = []

        for i in batch_paths:
            ct_scan_path, mask_path = ct_scans_paths[i], masks_paths[i]
            batch_ct_scans.append(load_ct_scan(ct_scan_path))
            batch_masks.append(load_mask(mask_path))

        yield np.array(batch_ct_scans), np.array(batch_masks)


# Example usage
ct_scans_paths = ['/home/mustafa/project/LUNA16/']  # Fill with actual paths
masks_paths = ['/home/mustafa/project/mask_folder/']  # Corresponding mask paths
batch_size = 32  # Adjust based on your GPU's memory
initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Add these metrics in model compilation
model.compile(optimizer=optimizer, loss='binary_crossentropy',
              metrics=[BinaryAccuracy(), Precision(), Recall(), TruePositives(), FalsePositives()])
train_gen = data_generator(ct_scans_paths, masks_paths, batch_size)
model.fit(train_gen, steps_per_epoch=len(ct_scans_paths) // batch_size, epochs=10)

# Load test data
test_ct_scan = load_ct_scan('/path/to/test_ct_scan.mhd')
test_mask = load_mask('/path/to/test_mask.npy')

# Predict
predicted_mask = model.predict(np.expand_dims(test_ct_scan, axis=0))

# Evaluate using appropriate metrics
# ...
