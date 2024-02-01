import pandas as pd
from datetime import datetime
import numpy as np
import os

from keras import regularizers
from keras.regularizers import l2
from scipy.ndimage import rotate
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Sequential
from sklearn.utils.class_weight import compute_class_weight


checkpoint_path = "/home/mustafa/project/checkpoints/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
base_dir = "/home/mustafa/project/LUNA16/cropped_nodules/"
# Load the CSV
nodules_csv = pd.read_csv("/home/mustafa/project/LUNA16/cropped_nodules.csv")
nodules = nodules_csv.rename(columns={'SN': 'ID'})

# Split abnormal and normal nodules
abnormal_nodules = nodules[nodules['state'] == 1]
normal_nodules = nodules[nodules['state'] == 0]

# Sample for validation
val_abnormal = abnormal_nodules.sample(frac=0.10)
print("val_abnormal = ", len(val_abnormal))
val_normal = normal_nodules.sample(frac=0.10)
print("val_abnormal = ", len(val_normal))

# Get the train split by dropping the validation samples
train_abnormal = abnormal_nodules.drop(val_abnormal.index)
print("train_abnormal = ", len(train_abnormal))
train_normal = normal_nodules.drop(val_normal.index)
print("train normal = ", len(train_normal))
# Convert IDs to paths
val_data = (val_abnormal['ID'].astype(str) + '.npy').tolist() + (val_normal['ID'].astype(str) + '.npy').tolist()
train_data = (train_abnormal['ID'].astype(str) + '.npy').tolist() + (train_normal['ID'].astype(str) + '.npy').tolist()

# Get the labels
val_labels = val_abnormal['state'].tolist() + val_normal['state'].tolist()
train_labels = train_abnormal['state'].tolist() + train_normal['state'].tolist()

print("x_val = ", len(val_data))
print("x_train = ", len(train_data))
print("y_val = ", len(val_labels))
print("y_train = ", len(train_labels))





# Assuming y_train is your array of labels
class_weights = compute_class_weight('balanced', classes=[0, 1], y=train_labels)

# Convert to dictionary format for TensorFlow
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}



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



class DataGenerator(Sequence):
# Learned from https://mahmoudyusof.github.io/facial-keypoint-detection/data-generator/
  def __init__(self, all_image_paths, labels, base_dir, output_size, shuffle=False, batch_size=10):
    """
    Initializes a data generator object
      :param csv_file: file in which image names and numeric labels are stored
      :param base_dir: the directory in which all images are stored
      :param output_size: image output size after preprocessing
      :param shuffle: shuffle the data after each epoch
      :param batch_size: The size of each batch returned by __getitem__
    """
    self.imgs = all_image_paths
    self.base_dir = base_dir
    self.output_size = output_size
    self.shuffle = shuffle
    self.batch_size = batch_size
    self.precomputed_indices = []
    self.labels = np.array(labels)
    self.on_epoch_end()





  # def precompute_sampling(self):
    # Precompute indices for the entire epoch
    # self.epoch_indices = np.random.choice(self.indices, size=len(self.imgs), p=self.sampling_probabilities, replace=True)
    # indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
  def on_epoch_end(self):
    self.indices = np.arange(len(self.imgs))
    #print("indices ", self.indices)
    if self.shuffle:
        np.random.shuffle(self.indices)
    # Precompute indices for all batches
    self.precomputed_indices = [self.indices[i:i+self.batch_size] for i in range(0, len(self.indices), self.batch_size)]

  def __len__(self):
    return int(len(self.imgs) / self.batch_size)

  def __getitem__(self, idx):
    ## Initializing Batch
    #  that one in the shape is just for a one channel images
    # if you want to use colored images you might want to set that to 3
    X = np.empty((self.batch_size, *self.output_size,1))
    # (x, y, h, w)
    y = np.empty((self.batch_size, 1))

    # Use precomputed indices
    current_indices = self.precomputed_indices[idx]
    for i, data_index in enumerate(current_indices):
      img_path = os.path.join(self.base_dir, self.imgs[data_index])
      img = np.load(img_path)
      # Adjust the shape of the image
      img = self.adjust_image_shape(img)

      X[i] = img
      y[i] = self.labels[data_index]  # Use numpy array for faster indexing

    return X, y
      # Augment if it's a minority class sample
      # label = self.labels[data_index]
      #if self.augment and label == 1:
          # Apply rotation-based augmentation
      #   rotation_angle = np.random.uniform(-2, 2)  # Example: rotate between -20 to 20 degrees
      #   img = rotate(img, angle=rotation_angle, axes=(0, 1), reshape=False)
  def adjust_image_shape(self, img):
        if len(img.shape) == 3:
            img = np.expand_dims(img, axis=3)

        if img.shape != (31,31,31,1):
            # Padding
            pad_width = [(0, max_sz - sz) for sz, max_sz in zip(img.shape, (31,31,31,1))]
            img = np.pad(img, pad_width, mode='constant')
            # Cropping
            crop = tuple(slice(sz) for sz in (31,31,31,1))
            img = img[crop]
        return img

'''
class DataGenerator(Sequence):
    def __init__(self, all_image_paths, labels, base_dir, output_size, shuffle=False, batch_size=10, is_training=True):
        self.imgs = all_image_paths
        self.base_dir = base_dir
        self.output_size = output_size
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.labels = labels
        self.is_training = is_training

        self.on_epoch_end()

        if self.is_training:
            # Calculate the number of abnormal (minority) samples
            self.num_abnormal_samples = sum(labels)

            # Define a probability for each sample to be chosen based on its class
            weights = [2 if label == 1 else 1 for label in labels]
            self.sampling_probabilities = np.array(weights) / sum(weights)
            self.sampling_probabilities /= self.sampling_probabilities.sum()

    def on_epoch_end(self):
        self.indices = np.arange(len(self.imgs))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(len(self.imgs) / self.batch_size)

    def __getitem__(self, idx):
        X = np.empty((self.batch_size, *self.output_size, 1))
        y = np.empty((self.batch_size, 1))

        if self.is_training:
            indices = np.random.choice(self.indices, size=self.batch_size, p=self.sampling_probabilities[self.indices],
                                       replace=True)
        else:
            indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        for i, data_index in enumerate(indices):
            img_path = os.path.join(self.base_dir, self.imgs[data_index])
            img = np.load(img_path)

            # Augment if it's a minority class sample and in training mode
            if self.is_training and self.labels[data_index] == 1:
                rotation_angle = np.random.uniform(-20, 20)
                img = rotate(img, angle=rotation_angle, axes=(0, 1), reshape=False)

            if len(img.shape) == 3:
                img = np.expand_dims(img, axis=3)

            if img.shape != (31, 31, 31, 1):
                pad_width = [(0, max_sz - sz) for sz, max_sz in zip(img.shape, (31, 31, 31, 1))]
                img = np.pad(img, pad_width, mode='constant')
                crop = tuple(slice(sz) for sz in (31, 31, 31, 1))
                img = img[crop]

            label = self.labels[data_index]
            X[i,] = img
            y[i] = label

        return X, y
'''

## Defining and training the model

model = Sequential([
  ## define the model's architecture
    layers.Conv3D(filters=16, kernel_size=3, padding='same',  kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001)),
    layers.LeakyReLU(),
    layers.BatchNormalization(),
    layers.Conv3D(filters=16, kernel_size=3, padding='same',  kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001)),
    layers.LeakyReLU(),
    layers.BatchNormalization(),
    layers.MaxPool3D(pool_size=2),
    layers.BatchNormalization(),
    layers.Conv3D(filters=32, kernel_size=3, padding='same',  kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001)),
    layers.LeakyReLU(),
    layers.BatchNormalization(),
    layers.Conv3D(filters=32, kernel_size=3, padding='same',  kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001)),
    layers.LeakyReLU(),
    layers.BatchNormalization(),
    layers.MaxPool3D(pool_size=2),
    layers.BatchNormalization(),

    layers.Conv3D(filters=64, kernel_size=3, padding='same',  kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001)),
    layers.LeakyReLU(),
    layers.BatchNormalization(),
    layers.Conv3D(filters=64, kernel_size=3, padding='same',  kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001)),
    layers.LeakyReLU(),
    layers.BatchNormalization(),
    layers.MaxPool3D(pool_size=2),
    layers.BatchNormalization(),

    #layers.Conv3D(filters=128, kernel_size=3, padding='same'),
    #layers.LeakyReLU(),
    #layers.BatchNormalization(),
    #layers.Conv3D(filters=128, kernel_size=3, padding='same'),
    #layers.LeakyReLU(),
    #layers.BatchNormalization(),
    #layers.MaxPool3D(pool_size=2),
    #layers.BatchNormalization(),

    layers.GlobalAveragePooling3D(),
    layers.Dense(units=128,
    kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.L2(1e-4),
    activity_regularizer=regularizers.L2(1e-5)),
    layers.LeakyReLU(),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(units=1, activation="sigmoid"),
])

batch_size=128
# Use augment=True for training so that minority class gets augmented
train_gen = DataGenerator(train_data, train_labels, base_dir, (31, 31, 31), batch_size=batch_size, shuffle=False)
# Use augment=False for validation so no augmentation happens
test_gen = DataGenerator(val_data, val_labels, base_dir, (31, 31, 31), batch_size=batch_size, shuffle=False)


#initial_learning_rate = 0.001
#lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
#)
epochs=50

initial_learning_rate = 0.1
final_learning_rate = 0.00001
learning_rate_decay_factor = (final_learning_rate / initial_learning_rate)**(1/epochs)
steps_per_epoch = int(len(train_data)/batch_size)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=initial_learning_rate,
                decay_steps=steps_per_epoch,
                decay_rate=learning_rate_decay_factor,
                staircase=True)

## compile the model first of course
opt = tf.keras.optimizers.experimental.AdamW(
    learning_rate=lr_schedule,
    beta_1=0.9,
    beta_2=0.99,
    epsilon=1e-06,

    weight_decay=0.004,
    ema_momentum= 0.99,
    name="AdamW",
)


# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Create a TensorBoard callback
log_dir="/home/mustafa/project/LUNA16/"+ datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
csv_logger = tf.keras.callbacks.CSVLogger('metrics_focal_lowmodel50_classWT_overs.csv')
#loss=keras.losses.BinaryCrossentropy(from_logits=True)
model.compile(optimizer=opt,
              loss = tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True, from_logits=False, gamma=5),
              metrics=['accuracy', 'AUC', tf.keras.metrics.SpecificityAtSensitivity(0.5), 'Precision', 'Recall',
                       'FalseNegatives', 'FalsePositives', 'TrueNegatives', 'TruePositives', BalancedAccuracy()])
model.build(input_shape= (128,None,None,None,1))
model.summary()
# now let's train the model
#EarlyStop = tf.keras.callbacks.EarlyStopping(monitor='precision', patience=4, restore_best_weights=True,)
history = model.fit(train_gen, validation_data = test_gen, epochs=epochs, shuffle = False , verbose = 1 , callbacks = [csv_logger, tensorboard_callback, cp_callback],
use_multiprocessing = True, class_weight=class_weight_dict, workers=12)
#,class_weight=None
model.save("focal_lowmodel50_classWT_overs.keras")