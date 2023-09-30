import numpy as np
import os
from tensorflow.python.keras.callbacks import TensorBoard
from keras import regularizers
#import matplotlib.pyplot as plt
from time import time
import tensorflow as tf
import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Sequential
import pandas
import random

new_model = tf.keras.models.load_model('gemerator_model')

# Check its architecture
new_model.summary()

# Evaluate the restored model
loss, acc = new_model.evaluate(val_data, val_labels, verbose=1)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

print(new_model.predict(val_data).shape)


