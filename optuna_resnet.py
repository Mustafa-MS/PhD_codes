# Creating the integrated code for hyperparameter tuning with Optuna based on the provided code snippets
import numpy as np
import os
import pandas
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import optuna
from tensorflow.keras import layers, models, Input
from tensorflow.keras.utils import Sequence


# [Assuming the data loading and preprocessing steps remain the same from the provided code]

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


class BalancedAccuracyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Metrics for training data
        TP_train = logs.get('true_positives')
        TN_train = logs.get('true_negatives')
        FP_train = logs.get('false_positives')
        FN_train = logs.get('false_negatives')

        # Metrics for validation data
        TP_val = logs.get('val_true_positives')
        TN_val = logs.get('val_true_negatives')
        FP_val = logs.get('val_false_positives')
        FN_val = logs.get('val_false_negatives')

        # Calculate balanced accuracy for training data
        sensitivity_train = TP_train / (TP_train + FN_train + tf.keras.backend.epsilon())
        specificity_train = TN_train / (TN_train + FP_train + tf.keras.backend.epsilon())
        balanced_accuracy_train = (sensitivity_train + specificity_train) / 2

        # Calculate balanced accuracy for validation data
        sensitivity_val = TP_val / (TP_val + FN_val + tf.keras.backend.epsilon())
        specificity_val = TN_val / (TN_val + FP_val + tf.keras.backend.epsilon())
        balanced_accuracy_val = (sensitivity_val + specificity_val) / 2

        # Log the balanced accuracies
        logs['balanced_accuracy'] = balanced_accuracy_train
        logs['val_balanced_accuracy'] = balanced_accuracy_val
        print(
            f"\nBalanced Accuracy for epoch {epoch}: Train: {balanced_accuracy_train:.4f}, Validation: {balanced_accuracy_val:.4f}")



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



# Modifying the res_block function to accept num_filters and dropout_rate as arguments
def res_block_optuna(x, filters, dropout_rate, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    if conv_shortcut:
        shortcut = layers.Conv3D(filters, 1, strides=stride, name=name + '_0_conv')(x)
    x = layers.Conv3D(filters, kernel_size, strides=stride, padding='same', name=name + '_1_conv')(x)
    x = layers.Conv3D(filters, kernel_size, padding='same', name=name + '_2_conv')(x)
    x = layers.Dropout(dropout_rate)(x)  # Adding dropout after the conv layer
    return x

# Objective function for Optuna
def objective(trial):
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    num_filters = trial.suggest_categorical("num_filters", [32, 64, 128])

    # [Assuming the model is defined using the res_block_optuna function]
    input_tensor = Input(shape=(31, 31, 31, 1))

    # Initial Convolution
    x = layers.Conv3D(32, 3, strides=2, padding="same")(input_tensor)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)

    # Residual Blocks
    x = res_block_optuna(x, filters=num_filters, dropout_rate=dropout_rate, name='block1')
    x = layers.MaxPool3D(2)(x)
    x = res_block_optuna(x, filters=num_filters, dropout_rate=dropout_rate, name='block2')
    x = layers.MaxPool3D(2)(x)
    x = res_block_optuna(x, filters=num_filters, dropout_rate=dropout_rate, name='block3')
    x = layers.MaxPool3D(2)(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    # Final dense layer
    output_tensor = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(input_tensor, output_tensor)

    # Data Generators
    train_gen = DataGenerator(train_image_paths, train_labels, base_dir, (31, 31, 31), batch_size=batch_size, shuffle=False)
    test_gen = DataGenerator(val_data, val_label, base_dir, (31, 31, 31), batch_size=batch_size, shuffle=False)

    # Learning Rate Schedule and Optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)
    opt = tf.keras.optimizers.experimental.AdamW(learning_rate=lr_schedule)

    # Model Compilation
    model.compile(optimizer=opt, loss=tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True, from_logits=True, gamma=5),
                  metrics=['Recall'])

    # Model Training
    history = model.fit(train_gen, validation_data=test_gen, epochs=5, shuffle=False, verbose=0)

    # Return validation loss for Optuna optimization
    val_recall = history.history["val_recall"][-1]
    return val_recall

# Setting up the Optuna study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# Printing the best trial details
print("Best trial:")
trial = study.best_trial
print("Value: {}".format(trial.value))
print("Params: ")
for key, value in trial.params.items():
    print("{}: {}".format(key, value))

