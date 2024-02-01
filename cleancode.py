from datetime import datetime
import numpy as np
import os
from scipy.ndimage import rotate
#import matplotlib.pyplot as plt
from time import time
import tensorflow as tf
#import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Sequential
import pandas
from sklearn.utils.class_weight import compute_class_weight


checkpoint_path = "/home/mustafa/project/checkpoints/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

nodules_csv = pandas.read_csv("/home/mustafa/project/LUNA16/cropped_nodules.csv")
base_dir = "/home/mustafa/project/LUNA16/cropped_nodules/"
all_image_paths = os.listdir(base_dir)
all_image_paths = sorted(all_image_paths,key=lambda x: int(os.path.splitext(x)[0]))
print("all imag paths ln= ",len(all_image_paths))
nodules = nodules_csv.rename(columns = {'SN':'ID'})

abnormal_nodules= nodules.loc[nodules['state'] == 1]
normal_nodules= nodules.loc[nodules['state'] == 0]

y = pandas.concat([abnormal_nodules.sample(frac=0.10), normal_nodules.sample(frac=0.10)])
h= y['ID'].tolist()
h_string= map(str, h)
val_data = [item + '.npy' for item in h_string]
train_image_paths =[item for item in all_image_paths if item not in val_data]
val_label = []
train_labels =[]


for index in range(len(nodules)):
    if str(index) +'.npy' not in val_data:
        train_labels.append(nodules['state'].iloc[index])


for num_nodule in val_data:
    val_label.append(nodules.iloc[int(num_nodule[:-4]),1])


print("x_val = ", len(val_data))
print("x_train = ", len(train_image_paths))
print("y_val = ", len(val_label))

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

'''
def balanced_accuracy(y_true, y_pred):
    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.round(tf.keras.backend.flatten(y_pred))

    tp = tf.keras.backend.sum(tf.keras.backend.cast((y_true * y_pred), 'float32'))
    tn = tf.keras.backend.sum(tf.keras.backend.cast(((1 - y_true) * (1 - y_pred)), 'float32'))

    fp = tf.keras.backend.sum(tf.keras.backend.cast(((1 - y_true) * y_pred), 'float32'))
    fn = tf.keras.backend.sum(tf.keras.backend.cast((y_true * (1 - y_pred)), 'float32'))

    sensitivity = tp / (tp + fn + tf.keras.backend.epsilon())
    specificity = tn / (tn + fp + tf.keras.backend.epsilon())

    return (sensitivity + specificity) / 2
'''
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
    self.labels = labels
    self.on_epoch_end()


    # Calculate the number of abnormal (minority) samples
    #self.num_abnormal_samples = sum(train_labels)

    # Define a probability for each sample to be chosen based on its class
    # This helps in oversampling the minority class
    #weights = [4 if label == 1 else 1 for label in train_labels]  # For instance, 4 times more likely for minority class
    #self.sampling_probabilities = np.array(weights) / sum(weights)
    #self.sampling_probabilities /= self.sampling_probabilities.sum()

  def on_epoch_end(self):
    self.indices = np.arange(len(self.imgs))
    #print("indices ", self.indices)
    if self.shuffle:
        np.random.shuffle(self.indices)

  def __len__(self):
    return int(len(self.imgs) / self.batch_size)

  def __getitem__(self, idx):
    ## Initializing Batch
    #  that one in the shape is just for a one channel images
    # if you want to use colored images you might want to set that to 3
    X = np.empty((self.batch_size, *self.output_size,1))
    # (x, y, h, w)
    y = np.empty((self.batch_size, 1))

    # get the indices of the requested batch
    indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
    # Adjust the way indices are selected for each batch based on sampling probabilities
    #indices = np.random.choice(self.indices, size=self.batch_size, p=self.sampling_probabilities[self.indices],
      #                         replace=True)

    for i, data_index in enumerate(indices):
      img_path = os.path.join(self.base_dir, self.imgs[data_index])
      img = np.load(img_path)

      # Augment if it's a minority class sample
      label = self.labels[data_index]
      #if label == 1:
          # Apply rotation-based augmentation
       #   rotation_angle = np.random.uniform(-20, 20)  # Example: rotate between -20 to 20 degrees
        #  img = rotate(img, angle=rotation_angle, axes=(0, 1), reshape=False)

      if len(img.shape) == 3:
          img = np.expand_dims(img, axis=3)

      if img.shape != (31,31,31,1):
            # Padding
        pad_width = [(0, max_sz - sz) for sz, max_sz in zip(img.shape, (31,31,31,1))]
        img = np.pad(img, pad_width, mode='constant')
            # Cropping
        crop = tuple(slice(sz) for sz in (31,31,31,1))
        img = img[crop]

      label = self.labels[data_index]
      X[i,] = img
      y[i] = label

    return X, y


## Defining and training the model

model = Sequential([
  ## define the model's architecture
    layers.Conv3D(filters=16, kernel_size=3, padding='same'),
    layers.LeakyReLU(),
    layers.BatchNormalization(),
    layers.Conv3D(filters=16, kernel_size=3, padding='same'),
    layers.LeakyReLU(),
    layers.BatchNormalization(),
    layers.MaxPool3D(pool_size=2),
    layers.BatchNormalization(),
    layers.Conv3D(filters=32, kernel_size=3, padding='same'),
    layers.LeakyReLU(),
    layers.BatchNormalization(),
    layers.Conv3D(filters=32, kernel_size=3, padding='same'),
    layers.LeakyReLU(),
    layers.BatchNormalization(),
    layers.MaxPool3D(pool_size=2),
    layers.BatchNormalization(),

    layers.Conv3D(filters=64, kernel_size=3, padding='same'),
    layers.LeakyReLU(),
    layers.BatchNormalization(),
    layers.Conv3D(filters=64, kernel_size=3, padding='same'),
    layers.LeakyReLU(),
    layers.BatchNormalization(),
    layers.MaxPool3D(pool_size=2),
    layers.BatchNormalization(),

    layers.Conv3D(filters=128, kernel_size=3, padding='same'),
    layers.LeakyReLU(),
    layers.BatchNormalization(),
    layers.Conv3D(filters=128, kernel_size=3, padding='same'),
    layers.LeakyReLU(),
    layers.BatchNormalization(),
    layers.MaxPool3D(pool_size=2),
    layers.BatchNormalization(),

    layers.GlobalAveragePooling3D(),
    layers.Dense(units=256),
    layers.LeakyReLU(),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(units=1, activation="sigmoid"),
])

batch_size=128

train_gen = DataGenerator(train_image_paths, train_labels, base_dir, (31, 31, 31), batch_size=batch_size, shuffle=True)
test_gen = DataGenerator(val_data, val_label, base_dir, (31, 31, 31), batch_size=batch_size, shuffle=False)


#initial_learning_rate = 0.001
#lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
#)
epochs=50

initial_learning_rate = 0.1
final_learning_rate = 0.00001
learning_rate_decay_factor = (final_learning_rate / initial_learning_rate)**(1/epochs)
steps_per_epoch = int(len(train_image_paths)/batch_size)

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

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Create a TensorBoard callback
log_dir="/home/mustafa/project/LUNA16/"+ datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
csv_logger = tf.keras.callbacks.CSVLogger('/home/mustafa/project/metrics/cleancode.csv')
#loss=keras.losses.BinaryCrossentropy(from_logits=True)
model.compile(optimizer=opt,
              loss = tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True, from_logits=True, gamma=5),
              metrics=['accuracy', 'AUC', tf.keras.metrics.SpecificityAtSensitivity(0.5), 'Precision', 'Recall',
                       'FalseNegatives', 'FalsePositives', 'TrueNegatives', 'TruePositives', BalancedAccuracy()])
model.build(input_shape= (128,None,None,None,1))
model.summary()
# now let's train the model
#EarlyStop = tf.keras.callbacks.EarlyStopping(monitor='precision', patience=4, restore_best_weights=True,)
history = model.fit(train_gen, validation_data = test_gen, epochs=epochs, shuffle = False , verbose = 1 , callbacks = [csv_logger, tensorboard_callback, cp_callback],
use_multiprocessing = True, workers=10)
#,class_weight=None
model.save("cleancode.keras")