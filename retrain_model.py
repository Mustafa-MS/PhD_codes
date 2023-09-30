from tensorflow import keras
import numpy as np
import os
import pandas
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import load_model


# It can be used to reconstruct the model identically.
reconstructed_model = load_model("3d_image_classification.h5")
print("Loaded model from disk")
# summarize model.
reconstructed_model.summary()

# The reconstructed model is already compiled and has retained the optimizer
# state, so training can resume:



nodules_csv = pandas.read_csv("/home/mustafa/project/LUNA16/cropped_nodules.csv")
base_dir = "/home/mustafa/project/LUNA16/cropped_nodules/"


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

print("train image done")

val_labels = nodules.loc[~nodules['ID'].isin(h)]
val_labels_list= val_labels['state'].tolist()
print("done val label")


print("x_val = ", len(val_data))
print("x_train = ", len(train_data))
print("y_val = ", len(val_labels_list))

val_data = val_data.tolist()
print("val to list")


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

train_gen = DataGenerator(train_data, train_label, base_dir, (31, 31, 31), batch_size=128, shuffle=True)
print("train gen")
val_gen = DataGenerator(val_data, val_labels_list, base_dir, (31, 31, 31), batch_size=128, shuffle=True)
print("test gen")



# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_precision", patience=10, mode='max', restore_best_weights=True)
# Train the model, doing validation at the end of each epoch
epochs = 1
history = reconstructed_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    shuffle=True,
    verbose=1,
    callbacks=[checkpoint_cb, early_stopping_cb],
)