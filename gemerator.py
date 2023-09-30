import numpy as np
import os
from tensorflow.python.keras.callbacks import TensorBoard
from keras import regularizers
import matplotlib.pyplot as plt
from time import time
import tensorflow as tf
import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Sequential
import pandas
from scipy.ndimage import zoom
import sklearn
import random
#from matplotlib import pyplot

nodules_csv = pandas.read_csv("/home/mustafa/project/LUNA16/cropped_nodules.csv")
#candidate_csv = pandas.read_csv("/home/mustafa/project/LUNA16/candidates_V3.csv")
base_dir = "/home/mustafa/project/LUNA16/cropped_nodules/"
all_image_paths = os.listdir(base_dir)
all_image_paths = sorted(all_image_paths,key=lambda x: int(os.path.splitext(x)[0]))
print("all imag paths ln= ",len(all_image_paths))
#all_image_paths = np.array(all_image_paths)
nodules = nodules_csv.rename(columns = {'SN':'ID'})

'''
abnormal_nodules= nodules.loc[nodules['state'] == 1]
normal_nodules= nodules.loc[nodules['state'] == 0]

y = pandas.concat([abnormal_nodules.sample(frac=0.90), normal_nodules.sample(frac=0.90)])
print("df done with 10% from each list")
h= y['ID'].tolist()
train_label = y['state'].tolist()
print("df converted to list")
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
#for index in range(len(nodules)):
#    if index not in h:
#        val_labels.append(nodules['state'].iloc[index])
'''

print("x_val = ", len(val_data))
print("x_train = ", len(train_data))
print("y_val = ", len(val_labels_list))

'''

#OLD code slow but working
#x = random.choices(all_image_paths, k=1000)
#abnormal_nodules= nodules.loc[nodules['state'] == 1]
#normal_nodules= nodules.loc[nodules['state'] == 0]

#why = pandas.concat([abnormal_nodules.sample(frac=0.10), normal_nodules.sample(frac=0.10)])
#x = [item for item in all_image_paths if item in y['ID']]
abnormal_nodules= nodules.loc[nodules['state'] == 1]
normal_nodules= nodules.loc[nodules['state'] == 0]

y = pandas.concat([abnormal_nodules.sample(frac=0.10), normal_nodules.sample(frac=0.10)])
h= y['ID'].tolist()
h_string= map(str, h)
val_data = [item + '.npy' for item in h_string]
print("val data len= ",len(val_data))
train_image_paths =[item for item in all_image_paths if item not in val_data]
print("val dddddata len= ",len(train_image_paths))
print("aftr hjgjgkjg labl")
val_label = []
print("aftr val labl")
#labels= nodules.iloc[:,1]
#
print("bfor train labl")
train_labels =[]
print("start hr")
for index in range(len(nodules)):
    if str(index) +'.npy' not in val_data:
        train_labels.append(nodules['state'].iloc[index])
print("finish train labls appn")
#for ID,state in nodules.iterrows():
#    if ID not in x:
#        labels.append(state)
#labels = labels.to_numpy()

for num_nodule in val_data:
    #print(file_path)
    #val_data.append(np.load(os.path.join(base_dir,num_nodule)))
    #val_data.append(num_nodule)
    val_label.append(nodules.iloc[int(num_nodule[:-4]),1])
    #print("x_val = ", len(val_data))
print("x_val = ", len(val_data))
print("x_train = ", len(train_image_paths))
print("y_val = ", len(val_label))

#print(z)
'''
def plot_nodule(nodule_crop):
    # Learned from ArnavJain
    # https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing
    f, plots = plt.subplots(int(nodule_crop.shape[0] / 4) + 1, 4, figsize=(10, 10))

    for z_ in range(nodule_crop.shape[0]):
        plots[int(z_ / 4), z_ % 4].imshow(nodule_crop[z_, :, :])

    # The last subplot has no image because there are only 19 images.
    plt.show()
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
      #print("img path 1", img_path)
      img = np.load(img_path)
      #while img.shape == (31,31,31):
      if len(img.shape) == 3:
          #print ("i = ", i)
          #print("img path ",img_path)
      #plot_nodule(img)
          img = np.expand_dims(img, axis=3)
      # Resize the image to (31,31,31)
      #zoom_factors = [31 / dim for dim in img.shape[:3]] + [1]  # keep the last dimension (channels) the same
      #img = zoom(img, zoom_factors, output_shape=(31,31,31,img.shape[3]))
        # If the image is still not of size (31,31,31,1), pad or crop it
      if img.shape != (31,31,31,1):
            # Padding
        pad_width = [(0, max_sz - sz) for sz, max_sz in zip(img.shape, (31,31,31,1))]
        img = np.pad(img, pad_width, mode='constant')
            # Cropping
        crop = tuple(slice(sz) for sz in (31,31,31,1))
        img = img[crop]
      #print("img shpe ", img.shape)
      ## this is where you preprocess the image
      ## make sure to resize it to be self.output_size
          #print("data index ", data_index)
      label = self.labels[data_index]
          #print("label ", label)
      #print("label ", label)
      ## if you have any preprocessing for
      ## the labels too do it here

      X[i,] = img
      y[i] = label
      #print("X ", X.shape)
      #print("y ", y.shape)
      #print("X ",X)
      #print("len label ", len(y))
      #print("labels ", y)
    return X, y


## Defining and training the model

model = Sequential([
  ## define the model's architecture
    layers.Conv3D(filters=32, kernel_size=3, activation="relu",padding='same'),
    #layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Conv3D(filters=32, kernel_size=3, activation="relu",padding='same'),
    #layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.MaxPool3D(pool_size=2),
    layers.BatchNormalization(),
    #layers.Flatten(),
    #layers.Dense(128, activation="relu"),
    #layers.Dense(1,activation="sigmoid"),


    layers.Conv3D(filters=64, kernel_size=3, activation="relu",padding='same'),
    #layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Conv3D(filters=64, kernel_size=3, activation="relu",padding='same'),
    #layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.MaxPool3D(pool_size=2),
    layers.BatchNormalization(),

    layers.Conv3D(filters=128, kernel_size=3, activation="relu",padding='same'),
    #layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Conv3D(filters=128, kernel_size=3, activation="relu",padding='same'),
    #layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.MaxPool3D(pool_size=2),
    layers.BatchNormalization(),

    layers.Conv3D(filters=256, kernel_size=3, activation="relu", padding='same'),
    layers.BatchNormalization(),
    layers.Conv3D(filters=256, kernel_size=3, activation="relu", padding='same'),
    layers.BatchNormalization(),
    layers.MaxPool3D(pool_size=2),
    layers.BatchNormalization(),

    layers.GlobalAveragePooling3D(),
    layers.Dense(units=512, activation="relu"),
    #layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(units=1),
])



train_gen = DataGenerator(train_image_paths, train_labels, base_dir, (31, 31, 31), batch_size=128, shuffle=False)
test_gen = DataGenerator(val_data, val_label, base_dir, (31, 31, 31), batch_size=128, shuffle=False)
    #print(train_gen.im)

## compile the model first of course
opt = tf.keras.optimizers.experimental.AdamW(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.99,
    epsilon=1e-06,
    weight_decay=0.004,
    ema_momentum= 0.99,
    name="AdamW",
)

# Create a TensorBoard callback
log_dir="/home/mustafa/project/LUNA16/{}".format(time())
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch = '5,15')


model.compile(optimizer=opt,
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy', 'binary_accuracy', 'AUC', tf.keras.metrics.SpecificityAtSensitivity(0.2), 'Precision', 'Recall', 'FalseNegatives', 'FalsePositives', 'TrueNegatives', 'TruePositives'])
model.build(input_shape= (128,None,None,None,1))
model.summary()
# now let's train the model
#EarlyStop = tf.keras.callbacks.EarlyStopping(monitor='precision', patience=4, restore_best_weights=True,)
history = model.fit(train_gen, validation_data = test_gen, epochs=1, shuffle = False , verbose = 1, callbacks = [tensorboard_callback],
use_multiprocessing = True)
model.save("gemerator_model_adamw")
# Evaluate the model on the test data using `evaluate`
#print("Evaluate on test data")
#results = model.evaluate(test_gen, batch_size=10)
#print("test loss, test acc:", results)
'''
fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()

for i, metric in enumerate(["accuracy", "loss"]):
    ax[i].plot(model.history.history[metric])
    ax[i].plot(model.history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])
    '''
# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
#print("Generate predictions for 3 samples")
#predictions = model.predict(val_data[:3])
#print("predictions shape:", predictions.shape)
# plot metrics
#pyplot.plot(history.history['accuracy'])
#pyplot.plot(history.history['loss'])
plt.interactive(False)
#pyplot.show()
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show(block=True)
'''
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
'''