import numpy as np
import pandas
import os
from keras.models import Sequential
#from my_classes import DataGenerator
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import keras
import tensorflow
from keras.utils import np_utils
from tensorflow.keras.utils import Sequence

# Parameters
params = {'dim': (19,19,19),
          'batch_size': 64,
          'n_classes': 2,
          'n_channels': 1,
          'shuffle': False}

# Datasets
nodules_csv = pandas.read_csv("/home/mustafa/project/LUNA16/cropped_nodules.csv")
#df = nodules_csv.values
partition = "/home/mustafa/project/LUNA16/cropped_nodules/"
#all_image_paths = files = sorted([os.path.join(partition, file) for file in os.listdir(partition)], key=os.path.getctime)
all_image_paths = [os.path.splitext(filename)[0] for filename in os.listdir(partition)]
#all_image_paths = np.array(all_image_paths)

#labels = df[:,1]
nodules = nodules_csv.rename(columns = {'SN':'ID'})
labels= nodules.iloc[:,1]
#labels = labels[:,1]
labels = labels.to_numpy()
#labels = labels.tolist()

#def batch_generator(image, label, batchsize):
#    N = len(image)
 #   i = 0
  #  print("n length ",N)
   # while True:
    #    img = np.load(image[i:i+batchsize])
     #   label[i:i+batchsize]
      #  print("label ", label)
       # print("image ", image)
        ##yield np.load(image[i:i+batchsize]), label[i:i+batchsize]
        #yield img, label
        #print("label ", label)
        #print("image ", image)
        #i = i + batchsize
        #if i + batchsize > N:
         #   i = 0
# Generators
class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=False):
        'Initialization'
        self.dim = dim
        #print(self.dim)
        self.batch_size = batch_size
        #print(self.batch_size)
        self.labels = labels
        #print(self.labels[:10])
        self.list_IDs = list_IDs
        #print(self.list_IDs[:10])
        self.n_channels = n_channels
        #print(self.n_channels)
        self.n_classes = n_classes
        #print(self.n_classes)
        self.shuffle = shuffle
        self.on_epoch_end()



    def __len__(self):
        'Denotes the number of batches per epoch'
        print("len ", int(np.floor(len(self.list_IDs) / self.batch_size)))
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        print("get item")
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        print("indexes ", indexes)
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        print("list ids temp", list_IDs_temp)
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        print("epoch ends ")
        self.indexes = np.arange(len(self.list_IDs))
        print("epoch len " ,len(self.list_IDs))
        print(self.indexes[:10])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            print("epoch shuffle ",self.indexes )

    def __data_generation(self, list_IDs_temp):
        print("__data generator ")
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('home/mustafa/project/LUNA16/cropped_nodules/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        #return X,y


training_generator = DataGenerator(all_image_paths, labels, batch_size=64, dim=(31,31,31), n_channels=1,
                n_classes=2, shuffle=False)
#validation_generator = DataGenerator(all_image_paths, labels, batch_size=64, dim=(19,19,19), n_channels=1,
  #               n_classes=2, shuffle=True)

# Design model
model = Sequential(
    [
    layers.Conv3D(filters=64, kernel_size=3, activation="relu"),
    layers.MaxPool3D(pool_size=2),
    layers.BatchNormalization(),

    layers.Conv3D(filters=64, kernel_size=3, activation="relu"),
    layers.MaxPool3D(pool_size=2),
    layers.BatchNormalization(),

    layers.Conv3D(filters=128, kernel_size=3, activation="relu"),
    layers.MaxPool3D(pool_size=2),
    layers.BatchNormalization(),

    layers.Conv3D(filters=256, kernel_size=3, activation="relu"),
    layers.MaxPool3D(pool_size=2),
    layers.BatchNormalization(),

    layers.GlobalAveragePooling3D(),
    layers.Dense(units=512, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(units=1, activation="sigmoid"),
        ])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(training_generator, epochs=5)
# Train model on dataset
#model.fit_generator(generator=training_generator,
 #                   epochs=5,
  #                  use_multiprocessing=True,
   #                 workers=6)