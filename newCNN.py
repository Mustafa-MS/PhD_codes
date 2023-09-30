import numpy as np
import os
import tensorflow as tf
from keras import Sequential
from keras.layers import Conv3D, BatchNormalization, LeakyReLU, Dropout, MaxPooling3D, Flatten, Dense, Activation

from tensorflow import keras
from tensorflow.keras import layers
import random

from scipy import ndimage



x_train = np.load('x_train_1_8.npy')
y_train = np.load('y_train_1_8.npy')
x_val = np.load('x_val_9.npy',  allow_pickle=True)
y_val = np.load('y_val_9.npy')



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
    #volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label



# Define data loaders.
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

batch_size = 2
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







model = Sequential()
model.add(Conv3D(64, kernel_size=(5, 5, 5), activation='linear',
                 kernel_initializer='glorot_uniform', input_shape=(128,128,64,1)))
model.add(BatchNormalization(center=True, scale=True))
model.add(LeakyReLU(.1))
model.add(Dropout(.25))
model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='linear',
                 kernel_initializer='glorot_uniform'))
model.add(BatchNormalization(center=True, scale=True))
model.add(LeakyReLU(.1))
model.add(MaxPooling3D(pool_size=(3, 3, 3)))
model.add(Dropout(.25))

model.add(Conv3D(256, kernel_size=(3, 3, 3), activation='linear',
                 kernel_initializer='glorot_uniform'))
model.add(BatchNormalization(center=True, scale=True))
model.add(LeakyReLU(.1))
model.add(Dropout(.25))
model.add(Conv3D(512, kernel_size=(3, 3, 3), activation='linear',
                 kernel_initializer='glorot_uniform'))
model.add(BatchNormalization(center=True, scale=True))
model.add(LeakyReLU(.1))
model.add(MaxPooling3D(pool_size=(3, 3, 3)))
model.add(Dropout(.25))

model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization(center=True, scale=True))
model.add(LeakyReLU(.1))
model.add(Dropout(.5))
model.add(Dense(512))
model.add(BatchNormalization(center=True, scale=True))
model.add(LeakyReLU(.1))
model.add(Dropout(.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


#model = get_model(width=128, height=128, depth=64)




# Compile model.
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
)
model.summary()
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
    )
