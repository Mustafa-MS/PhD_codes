import glob
import SimpleITK as sitk
from scipy import ndimage
import numpy as np
import os
from tensorflow.python.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Sequential
import pandas
from tensorflow import keras
from tensorflow.keras import layers


nodules_csv = pandas.read_csv("/home/mustafa/project/LUNA16/cropped_nodules.csv")
base_dir = "/home/mustafa/project/LUNA16/cropped_nodules/"

luna_test_path = '/home/mustafa/project/LUNA16/subset9/'
test_file_list = glob.glob(luna_test_path + '/*.mhd')

all_image_paths = os.listdir(base_dir)
all_image_paths = sorted(all_image_paths,key=lambda x: int(os.path.splitext(x)[0]))
all_image_paths = np.array(all_image_paths)
nodules = nodules_csv.rename(columns = {'SN':'ID'})

abnormal_nodules= nodules.loc[nodules['state'] == 1]
normal_nodules= nodules.loc[nodules['state'] == 0]


y = pandas.concat([abnormal_nodules.sample(frac=0.90), normal_nodules.sample(frac=0.90)])
print("df done with 10% from each list done")
h= y['ID'].tolist()
train_label = y['state'].tolist()
print("df converted to list done")
print(" done updating the code")
h_string= map(str, h)
print("list converted from int to str")
train_data = [item + '.npy' for item in h_string]
print ("validata done with npy")


filtered_array= np.isin(all_image_paths,train_data,invert=True)
val_data= all_image_paths[filtered_array]
#
print("val data length= ",len(val_data))
print("train data length= ",len(train_data))
#train_image_paths =[item for item in all_image_paths if item not in val_data]
print("train image done")

#val_labels =[item for item in nodules if int(item['ID']) not in h]
val_labels = nodules.loc[~nodules['ID'].isin(h)]
val_labels_list= val_labels['state'].tolist()
print("done val label")
'''
for index in range(len(nodules)):
    if index not in h:
        val_labels.append(nodules['state'].iloc[index])
'''

print("x_val = ", len(val_data))
print("x_train = ", len(train_data))
print("y_val = ", len(val_labels_list))

val_data = val_data.tolist()
print("val to list")

window_val =[]


def read_mhd_file(filepath):
    """Read and load volume"""
    # Read file
    scan = sitk.ReadImage(filepath)
    scan = sitk.GetArrayFromImage(scan)
    scan = np.moveaxis(scan, 0, 2)
    return scan


def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[2]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = np.flip(img , axis=2)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_mhd_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    #mem()
    return volume




for file_path in test_file_list:
    #print(file_path)
    window_val.append(process_scan(file_path))
    print("window_val = ", len(window_val))







def plot_nodule(nodule_crop):
    # Learned from ArnavJain
    # https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing
    f, plots = plt.subplots(int(nodule_crop.shape[0] / 4) + 1, 4, figsize=(10, 10))

    for z_ in range(nodule_crop.shape[0]):
        plots[int(z_ / 4), z_ % 4].imshow(nodule_crop[z_, :, :])

    # The last subplot has no image because there are only 19 images.
    plt.show()

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

    if self.shuffle:
      np.random.shuffle(self.indices)

  def __len__(self):
    return int(len(self.imgs) / self.batch_size)

  def __getitem__(self, idx):
    ## Initializing Batch
    #  that one in the shape is just for a one channel images
    #print("getitem")
    # if you want to use colored images you might want to set that to 3
    X = np.empty((self.batch_size, *self.output_size,1))
    # (x, y, h, w)
    y = np.empty((self.batch_size, 1))

    # get the indices of the requested batch
    indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]

    for i, data_index in enumerate(indices):
      img_path = os.path.join(self.base_dir,
                  self.imgs[data_index])

      img = np.load(img_path)
      while img.shape == (31,31,31):

          img = np.expand_dims(img, axis=3)

      ## this is where you preprocess the image
      ## make sure to resize it to be self.output_size
          #print("data index ", data_index)
          label = self.labels[data_index]

      ## if you have any preprocessing for
      ## the labels too do it here

          X[i,] = img
          y[i] = label

    return X, y


def get_model(width=31, height=31, depth=31):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", padding='same')(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", padding='same')(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu", padding='same')(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu", padding='same')(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model

train_gen = DataGenerator(train_data, train_label, base_dir, (31, 31, 31), batch_size=128, shuffle=True)
print("train gen")
val_gen = DataGenerator(val_data, val_labels_list, base_dir, (31, 31, 31), batch_size=128, shuffle=True)
print("test gen")

# Build model.
model = get_model(width=31, height=31, depth=31)
model.summary()

# Compile model.
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm= 0.001),
    metrics=['binary_accuracy', 'Precision', 'Recall', 'FalseNegatives', 'FalsePositives', 'TrueNegatives', 'TruePositives'],
)

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_precision", patience=10, mode='max', restore_best_weights=True)
# Train the model, doing validation at the end of each epoch
epochs = 1
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    shuffle=True,
    verbose=1,
    callbacks=[checkpoint_cb, early_stopping_cb],
)



def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for z in range(0, image.shape[2], stepSize[2]):
        for y in range(0, image.shape[1], stepSize[1]):
            for x in range(0, image.shape[0], stepSize[0]):
                # yield the current window
                yield (x, y, z, image[x:x + windowSize[0], y:y + windowSize[1], z:z + windowSize[2]])

# parameters
window_size = (31, 31, 31)  # The size of the scanning window, change to suit your needs
step_size = (10, 10, 20)  # The amount of pixels the window moves at each step, change to suit your needs

# get your image
#image = np.load('your_image.npy')  # Load a 3D numpy array
image = window_val[1]

# slide the window over the image
for (x, y, z, window) in sliding_window(image, step_size, window_size):
    # Here you can apply your 3D CNN to the window, e.g.:
    if window.shape[0] != window_size[0] or window.shape[1] != window_size[1] or window.shape[2] != window_size[2]:
        continue  # Skip if window doesn't meet size requirements (at the edges)

    window = np.expand_dims(window, axis=0)  # Add an extra dimension for the batch size
    window = np.expand_dims(window, axis=4)  # Add an extra dimension for grayscale
    prediction = model.predict(window)  # Apply your model to the window
    print(f'Prediction for window at position {(x, y, z)}: {prediction}')


plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()

for i, metric in enumerate(["binary_accuracy", "loss"]):
    ax[i].plot(history.history[metric])
    ax[i].plot(history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])