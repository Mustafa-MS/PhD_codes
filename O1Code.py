import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

# Load labels from CSV
labels_df = pd.read_csv('/home/mustafa/project/dataset/cropped_nodules.csv')

# Prepare file paths and labels
image_dir = '/home/mustafa/project/dataset/cropped_nodules/'  # directory where .npy files are stored
file_paths = [os.path.join(image_dir, f'{i}.npy') for i in labels_df['SN']]
#labels = labels_df['state'].values
labels = labels_df['state'].values.astype('float32')
# Convert to numpy arrays
file_paths = np.array(file_paths)
labels = np.array(labels)

# Stratified train-test split
train_file_paths, val_file_paths, train_labels, val_labels = train_test_split(
  file_paths,
  labels,
  test_size=0.2,
  stratify=labels,
  random_state=42
)
print ("train ln", len(train_labels))
print("val ln", len(val_labels))
# Ensure labels are float32
train_labels = train_labels.astype('float32')
val_labels = val_labels.astype('float32')

# Compute class weights
class_weights = class_weight.compute_class_weight(
  class_weight='balanced',
  classes=np.unique(labels),
  y=labels
)
class_weights_dict = {i : class_weights[i] for i in range(len(class_weights))}
print(class_weights_dict)

# Create datasets
batch_size = 128

def load_npy(file_path, label):
  image = np.load(file_path.numpy().decode('utf-8'))
  image = np.expand_dims(image, axis=-1)
  image = image.astype('float32') / 255.0
  return image, label

def load_npy_wrapper(file_path, label):
  image, label = tf.py_function(func=load_npy, inp=[file_path, label], Tout=(tf.float32, tf.float32))
  image.set_shape([31, 31, 31, 1])
  label.set_shape([])
  return image, label

train_dataset = tf.data.Dataset.from_tensor_slices((train_file_paths, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=1000)
train_dataset = train_dataset.map(load_npy_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_file_paths, val_labels))
val_dataset = val_dataset.map(load_npy_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

def create_3d_cnn_model(input_shape):
  inputs = keras.Input(shape=input_shape)

  x = layers.Conv3D(32, kernel_size=3, activation='relu')(inputs)
  x = layers.MaxPooling3D(pool_size=2)(x)

  x = layers.Conv3D(64, kernel_size=3, activation='relu')(x)
  x = layers.MaxPooling3D(pool_size=2)(x)

  x = layers.Conv3D(128, kernel_size=3, activation='relu')(x)
  x = layers.Flatten()(x)

  x = layers.Dense(128, activation='relu')(x)
  x = layers.Dropout(0.5)(x)
  outputs = layers.Dense(1, activation='sigmoid')(x)

  model = keras.Model(inputs, outputs, name='3d_cnn_model')
  return model

input_shape = (31, 31, 31, 1)
model = create_3d_cnn_model(input_shape)
model.summary()

model.compile(
  optimizer=keras.optimizers.Adam(learning_rate=1e-4),
  loss='binary_crossentropy',
  metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

epochs = 10  # Adjust based on your needs

# Include callbacks if needed
callbacks = [
  keras.callbacks.ModelCheckpoint('3d_cnn_model.keras', save_best_only=True),
  keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
]

history = model.fit(
  train_dataset,
  validation_data=val_dataset,
  epochs=epochs,
  callbacks=callbacks,
  class_weight=class_weights_dict
)

# Evaluate on validation set
val_loss, val_accuracy, val_auc = model.evaluate(val_dataset)
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_accuracy}')
print(f'Validation AUC: {val_auc}')
