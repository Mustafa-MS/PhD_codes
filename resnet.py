from datetime import datetime
import numpy as np
import os
from tensorflow.keras import layers, models, Input
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
import tensorflow_addons as tfa
import pandas
from sklearn.utils.class_weight import compute_class_weight

checkpoint_path = "/home/mustafa/project/checkpoints/resnet100/{epoch:02d}-{val_loss:.2f}.keras"
checkpoint_dir = os.path.dirname(checkpoint_path)

nodules_csv = pandas.read_csv("/home/mustafa/project/LUNA16/cropped_nodules.csv")
base_dir = "/home/mustafa/project/LUNA16/cropped_nodules/"
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


'''
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
'''
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


# Constants for regularization
l2_lambda = 0.0001  # L2 regularization constant
dropout_rate = 0.5  # Dropout rate


def res_block(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    """A residual block."""
    if conv_shortcut:
        shortcut = layers.Conv3D(filters, 1, strides=stride,kernel_regularizer=tf.keras.regularizers.l2(l2_lambda),
                                 name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv3D(filters, kernel_size, strides=stride, padding='same', kernel_regularizer=tf.keras.regularizers.l2(l2_lambda),
                      name=name + '_1_conv')(x)
    x = layers.BatchNormalization(name=name + '_1_bn')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv3D(filters, kernel_size, padding='same', kernel_regularizer=tf.keras.regularizers.l2(l2_lambda),
                      name=name + '_2_conv')(x)
    x = layers.BatchNormalization(name=name + '_2_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.LeakyReLU(name=name + '_out')(x)
    return x


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

    for i, data_index in enumerate(indices):
      img_path = os.path.join(self.base_dir, self.imgs[data_index])
      img = np.load(img_path)

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

input_tensor = Input(shape=(None, None, None, 1))

# Initial Convolution
x = layers.Conv3D(32, 3, strides=2, padding="same", kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))(input_tensor)
x = layers.LeakyReLU()(x)
x = layers.BatchNormalization()(x)

# Residual Blocks
x = res_block(x, 32, name='block1')
x = layers.MaxPool3D(2)(x)
x = res_block(x, 64, name='block2')
x = layers.MaxPool3D(2)(x)
x = res_block(x, 128, name='block3')
x = layers.MaxPool3D(2)(x)

x = layers.GlobalAveragePooling3D()(x)
x = layers.Dense(256)(x)
x = layers.LeakyReLU()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(dropout_rate)(x)

# Final dense layer
output_tensor = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(input_tensor, output_tensor)

batch_size=128

train_gen = DataGenerator(train_data, train_labels, base_dir, (31, 31, 31), batch_size=batch_size, shuffle=False)
test_gen = DataGenerator(val_data, val_labels, base_dir, (31, 31, 31), batch_size=batch_size, shuffle=False)


#initial_learning_rate = 0.001
#lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
#)
epochs=100

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
csv_logger = tf.keras.callbacks.CSVLogger('metrics_resnet100.csv')

#loss=keras.losses.BinaryCrossentropy(from_logits=True)
model.compile(optimizer=opt,
              loss = tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True, from_logits=True, gamma=5),
              metrics=[BalancedAccuracy(), 'AUC' ,tfa.metrics.F1Score(num_classes=1, threshold=0.5), tf.keras.metrics.SpecificityAtSensitivity(0.5), 'Precision', 'Recall',
                     'TruePositives', 'FalsePositives', 'FalseNegatives', 'TrueNegatives', 'accuracy'])
model.build(input_shape= (batch_size,None,None,None,1))
model.summary()
# now let's train the model
#EarlyStop = tf.keras.callbacks.EarlyStopping(monitor='precision', patience=4, restore_best_weights=True,)
history = model.fit(train_gen, validation_data = test_gen, epochs=epochs, shuffle = False , verbose = 1 , callbacks = [csv_logger, cp_callback],
use_multiprocessing = True, class_weight=class_weight_dict)
#,class_weight=None
model.save("resnet100.keras")