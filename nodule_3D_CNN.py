import gc
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
#from collections import deque
#from tensorflow import keras
#import tensorflow_addons as tfa
#from tensorflow.keras import layers
#import random
import pandas
import pathlib
#import tensorflow.experimental.numpy as tnp
#tnp.experimental_enable_numpy_behavior()


#from scipy import ndimage
#import objgraph
nodules_path = "/home/mustafa/project/LUNA16/cropped_nodules/"
nodules_csv = pandas.read_csv("/home/mustafa/project/LUNA16/cropped_nodules.csv")
#all_image_paths = [f for f in os.listdir(nodules_path) if os.path.isfile( os.path.join(nodules_path, f) )]
all_image_paths = files = sorted([os.path.join(nodules_path, file) for file in os.listdir(nodules_path)], key=os.path.getctime)




def plot_nodule(nodule_crop):
    # Learned from ArnavJain
    # https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing
    f, plots = plt.subplots(int(nodule_crop.shape[0] / 4) + 1, 4, figsize=(10, 10))

    for z_ in range(nodule_crop.shape[0]):
        plots[int(z_ / 4), z_ % 4].imshow(nodule_crop[z_, :, :])

    # The last subplot has no image because there are only 19 images.
    plt.show()


df = nodules_csv[:-1067]
df = df[['state']]


def load_image(path):
    image = tf.io.read_file(path)
    return image

ds = tf.data.Dataset.from_tensor_slices((all_image_paths, df))

# The tuples are unpacked into the positional arguments of the mapped function
def load_from_path_label(path, label):
    return load_image(path), label

image_label_ds = ds.map(load_from_path_label)

'''
#This will print image and label
for image_raw, label_text in image_label_ds.take(1):
  print(repr(image_raw.numpy()[:100]))
  print()
  print(label_text.numpy())
  plot_nodule(image_raw.numpy())
'''

neg, pos = np.bincount(df['state'])
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))
#train_df, test_df = train_test_split(df, test_size=0.2)
train_df, test_df = train_test_split(tf.data.Dataset.from_tensor_slices((all_image_paths, df)), test_size=0.2)
#train_df, val_df = train_test_split(train_df, test_size=0.2)


'''
This test takes long time
dataset = tf.data.Dataset.from_tensor_slices(([f for f in os.listdir(nodules_path) if os.path.isfile(np.load (os.path.join(nodules_path, f)))], df))
#, iterator=True, chunksize=1000
#nodules_csv = pandas.read_csv("/home/mustafa/project/LUNA16/cropped_nodules_2.csv")
'''

'''
def map_func(name):
  feature = np.load(os.path.join(nodules_path,name.decode("utf-8")))
  return feature.astype(np.float32)

feature_paths = [f for f in os.listdir(nodules_path) if os.path.isfile( os.path.join(nodules_path, f) )]
x = [w for w in sorted(feature_paths)]
feature_paths.sort(key = str)
dataset = tf.data.Dataset.from_tensor_slices(feature_paths)

# Use map to load the numpy files in parallel
dataset = dataset.map(lambda item: tf.numpy_function(
          map_func, [item], tf.float32),
          num_parallel_calls=tf.data.AUTOTUNE)


iterator = iter(dataset)
next_element = iterator.get_next()
plt.imshow(next_element[1,:,:])
plt.show()
'''

'''
global positive
global negative
positive = 0
negative = 0
x_val = []
#x_val = deque()
x_train = []
#x_train = deque()
y_train = []
#y_train = deque()
y_val = []
#y_val = deque()
print(gc.isenabled())
#gc.disable()
#print(gc.isenabled())
#for nodule in nodules_csv.itertuples():
def train_test (SN, state):
#chunksize = 10 ** 4
#with pandas.read_csv("/home/mustafa/project/LUNA16/cropped_nodules_2.csv", chunksize=chunksize) as reader:

#for chunk in nodules_csv:
    #print(chunk)
#    while len(x_train) <= 50000 :
#        for index, nodule in chunk.iterrows():
    global negative
    global positive
    if state == 1 and positive <= 700 and len(x_val) <= 2000 :
        #gc.disable()
        positive += 1
        x_val_img = str(SN) + ".npy"
        x_val.append(np.load(os.path.join(nodules_path,x_val_img)))
        y_val.append(state)
        #print("positive= ", positive)
        #print("x_val= ", x_val_img)
        print("x_val len= ", len(x_val))
        #print(gc.isenabled())
        #gc.enable()
    # elif nodule.state == 1 and positive > 700:
    #     #gc.disable()
    #     x_train_img = str(nodule.SN) + ".npy"
    #     x_train.append(np.load(os.path.join(nodules_path,x_train_img)))
    #     y_train.append(nodule.state)
    #     print("x_train len= ", len(x_train))
    #     #print(gc.isenabled())
    #     #gc.enable()
    #     #print("x_train 1= ", x_train_img)
    elif state == 0 and negative <= 1300 and len(x_val) <= 2000:
        #gc.disable()
        x_val_img = str(SN) + ".npy"
        negative += 1
        x_val.append(np.load(os.path.join(nodules_path,x_val_img)))
        y_val.append(state)
        #print("negative= ", negative)
        #print("x_val= ", x_val_img)
        print("x_val len= ", len(x_val))
        print("Size of list1: " + str(sys.getsizeof(x_val)) + "bytes")
        #gc.enable()
    else:
        #gc.disable()
        if len(x_train) % 10000 == 0:
            gc.collect()
            print("gc collected")
        x_train_img = str(SN) + ".npy"
        x_train.append(np.load(os.path.join(nodules_path,x_train_img)))
        y_train.append(state)
        print("x_train len= ", len(x_train))
        print("Size of list1: " + str(sys.getsizeof(x_train)) + "bytes")
        #gc.enable()
        #print("x_train 0= ", x_train_img)
        return x_train, y_train, x_val, y_val



#import pdb; pdb.set_trace()
#print(gc.isenabled())
#gc.enable()
#print(gc.isenabled())
from guppy import hpy
hp = hpy()
before = hp.heap()
sn = nodules_csv['SN'].values
stt = nodules_csv['state'].values
vfunc = np.vectorize(train_test)
x_train, y_train, x_val, y_val = vfunc(sn,stt)
#x_train, y_train, x_val, y_val = train_test(nodules_csv['SN'].values, nodules_csv['state'].values)
after = hp.heap()
leftover = after - before
print(leftover)
print("finished")
@tf.function
def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


# Define data loaders.
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

batch_size = 1
# Augment the on the fly during training.
train_dataset = (
    train_loader.shuffle(len(x_train))
        .map(train_preprocessing)
        .batch(batch_size)
        .prefetch(2)
)
# Only rescale.
validation_dataset = (
    validation_loader.shuffle(len(x_val))
        .map(validation_preprocessing)
        .batch(batch_size)
        .prefetch(2)
)

import matplotlib.pyplot as plt

data = train_dataset.take(1)
images, labels = list(data)[0]
images = images.numpy()
image = images[0]
print("Dimension of the CT scan is:", image.shape)
plt.imshow(np.squeeze(image[:, :, 30]), cmap="gray")


def plot_slices(num_rows, num_columns, width, height, data):
    """Plot a montage of 20 CT slices"""
    data = np.rot90(np.array(data))
    data = np.transpose(data)
    data = np.reshape(data, (num_rows, num_columns, width, height))
    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )
    for i in range(rows_data):
        for j in range(columns_data):
            axarr[i, j].imshow(data[i][j], cmap="gray")
            axarr[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()


# Visualize montage of slices.
# 4 rows and 10 columns for 100 slices of the CT scan.
plot_slices(4, 10, 128, 128, image[:, :, :40])


def get_model(width=128, height=128, depth=64):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


# Build model.
model = get_model(width=128, height=128, depth=64)
model.summary()

# Compile model.
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(
    loss=tfa.losses.SigmoidFocalCrossEntropy(),
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
)

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

# Train the model, doing validation at the end of each epoch
epochs = 100
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    callbacks=[checkpoint_cb, early_stopping_cb],
)

fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()

for i, metric in enumerate(["acc", "loss"]):
    ax[i].plot(model.history.history[metric])
    ax[i].plot(model.history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])

# Load best weights.
model.load_weights("3d_image_classification.h5")
prediction = model.predict(np.expand_dims(x_val[0], axis=0))[0]
scores = [1 - prediction[0], prediction[0]]

class_names = ["normal", "abnormal"]
for score, name in zip(scores, class_names):
    print(
        "This model is %.2f percent confident that CT scan is %s"
        % ((100 * score), name)
    )'''