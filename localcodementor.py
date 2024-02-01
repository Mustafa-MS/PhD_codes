import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import glob
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.losses import BinaryFocalCrossentropy
from scipy.ndimage import zoom
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import backend as K

# Load truth table
truth_table = pd.read_csv('/home/mustafa/project/LUNA16/cropped_nodules.csv')

# Splitting based on class
nodules = truth_table[truth_table['state'] == 1]
non_nodules = truth_table[truth_table['state'] == 0]

# Train-test split (80-20)
train_nodules, val_nodules = train_test_split(nodules, test_size=0.2, random_state=42)
train_non_nodules, val_non_nodules = train_test_split(non_nodules, test_size=0.2, random_state=42)

# Combine nodules and non-nodules for train and validation sets
train_df = pd.concat([train_nodules, train_non_nodules])
val_df = pd.concat([val_nodules, val_non_nodules])

# Shuffle the data
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Assuming y_train is your array of labels
class_weights = compute_class_weight('balanced', classes=[0, 1], y=train_df['state'].values)

# Convert to dictionary format for TensorFlow
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print("class_wheight= ", class_weight_dict)


def pad_image(image, target_shape):
    # Calculate the padding widths
    pad_widths = [(0, t - s) for s, t in zip(image.shape, target_shape)]

    # Pad the image
    padded_image = np.pad(image, pad_widths, mode='constant', constant_values=0)
    return padded_image

'''
def precision_threshold(threshold=0.5, name='precision_custom'):
    def precision(y_true, y_pred):
        """Precision metric.
        Computes the precision over the whole batch using threshold_value.
        """
        threshold_value = threshold
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        # count the predicted positives
        predicted_positives = K.sum(y_pred)
        # Get the precision ratio
        precision_ratio = true_positives / (predicted_positives + K.epsilon())
        return precision_ratio

    precision.__name__ = name  # Set a unique name for the metric
    return precision
'''

class PrecisionThreshold(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, name='precision_threshold', **kwargs):
        super(PrecisionThreshold, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.predicted_positives = self.add_weight(name='pp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(tf.greater_equal(y_pred, self.threshold), tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        true_positives = tf.reduce_sum(y_true * y_pred)
        predicted_positives = tf.reduce_sum(y_pred)

        self.true_positives.assign_add(true_positives)
        self.predicted_positives.assign_add(predicted_positives)

    def result(self):
        precision = tf.divide(self.true_positives, self.predicted_positives + tf.keras.backend.epsilon())
        return precision

    def reset_state(self):
        self.true_positives.assign(0.0)
        self.predicted_positives.assign(0.0)

'''
def recall_threshold(threshold=0.5, name='recall_custom'):
    def recall(y_true, y_pred):
        """Recall metric.
        Computes the recall over the whole batch using threshold_value.
        """
        threshold_value = threshold
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        # Compute the number of positive targets.
        possible_positives = K.sum(K.clip(y_true, 0, 1))
        recall_ratio = true_positives / (possible_positives + K.epsilon())
        return recall_ratio

    recall.__name__ = name  # Set a unique name for the metric
    return recall
'''

class RecallThreshold(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, name='recall_threshold', **kwargs):
        super(RecallThreshold, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.possible_positives = self.add_weight(name='pp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(tf.greater_equal(y_pred, self.threshold), tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        true_positives = tf.reduce_sum(y_true * y_pred)
        possible_positives = tf.reduce_sum(y_true)

        self.true_positives.assign_add(true_positives)
        self.possible_positives.assign_add(possible_positives)

    def result(self):
        recall = tf.divide(self.true_positives, self.possible_positives + tf.keras.backend.epsilon())
        return recall

    def reset_state(self):
        self.true_positives.assign(0.0)
        self.possible_positives.assign(0.0)


# Step 1: Define the Data Generator

class NoduleDataGenerator(Sequence):
    def __init__(self, df, image_dir, batch_size, dim, n_channels, shuffle=True):
        self.df = df
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.indices = self.df.index.tolist()
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.df.iloc[k]['SN'] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, list_IDs_temp):
        # Generates data containing batch_size samples
        X = np.empty((len(list_IDs_temp), *self.dim, self.n_channels))
        y = np.empty((len(list_IDs_temp)), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Load sample
            # Load and reshape sample
            img = np.load(os.path.join(self.image_dir, str(ID) + '.npy'))

            if len(img.shape) == 3:
                img = np.expand_dims(img, axis=-1)

            if img.shape != (31, 31, 31, 1):
                # Padding
                pad_width = [(0, max_sz - sz) for sz, max_sz in zip(img.shape, (31, 31, 31, 1))]
                img = np.pad(img, pad_width, mode='constant')
                # Cropping
                crop = tuple(slice(sz) for sz in (31, 31, 31, 1))
                img = img[crop]
            # Store class
            X[i,] = img
            y[i] = self.df.loc[self.df['SN'] == ID]['state'].iloc[0]

        return X, y


# Step 2: Instantiate the Data Generator

batch_size = 128  # You can adjust this based on your system's capabilities
dim = (31, 31, 31)  # Dimensions of your input images
n_channels = 1  # Number of channels (1 for grayscale, 3 for RGB)

# Create generators
training_generator = NoduleDataGenerator(train_df, '/home/mustafa/project/LUNA16/cropped_nodules/', batch_size, dim,
                                         n_channels)
print("trining_gen")
validation_generator = NoduleDataGenerator(val_df, '/home/mustafa/project/LUNA16/cropped_nodules/', batch_size, dim,
                                           n_channels)
print("validating_gen")


model = Sequential([
  ## define the model's architecture
    layers.Conv3D(filters=16, kernel_size=3, padding='same'),
    layers.LeakyReLU(),
    layers.BatchNormalization(),
    layers.Conv3D(filters=16, kernel_size=3, padding='same'),
    layers.LeakyReLU(),
    layers.BatchNormalization(),
    layers.MaxPool3D(pool_size=2),
    layers.BatchNormalization(),
    layers.Conv3D(filters=32, kernel_size=3, padding='same'),
    layers.LeakyReLU(),
    layers.BatchNormalization(),
    layers.Conv3D(filters=32, kernel_size=3, padding='same'),
    layers.LeakyReLU(),
    layers.BatchNormalization(),
    layers.MaxPool3D(pool_size=2),
    layers.BatchNormalization(),

    layers.Conv3D(filters=64, kernel_size=3, padding='same'),
    layers.LeakyReLU(),
    layers.BatchNormalization(),
    layers.Conv3D(filters=64, kernel_size=3, padding='same'),
    layers.LeakyReLU(),
    layers.BatchNormalization(),
    layers.MaxPool3D(pool_size=2),
    layers.BatchNormalization(),

    layers.Conv3D(filters=128, kernel_size=3, padding='same'),
    layers.LeakyReLU(),
    layers.BatchNormalization(),
    layers.Conv3D(filters=128, kernel_size=3, padding='same'),
    layers.LeakyReLU(),
    layers.BatchNormalization(),
    layers.MaxPool3D(pool_size=2),
    layers.BatchNormalization(),

    layers.GlobalAveragePooling3D(),
    layers.Dense(units=256),
    layers.LeakyReLU(),
    layers.BatchNormalization(),
    #layers.Dropout(0.4),
    layers.Dense(units=1, activation="sigmoid"),
])




# Weighted binary cross-entropy to address class imbalance
weights = {0: 1, 1: (753791 / 1186)}  # Adjust weights as appropriate for your dataset
#loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=False) this loss function have higher FP
loss_function2 = tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True, from_logits=False, # this loss have lower FP when class balancing is true
    gamma=5.0)
# Define callbacks for checkpoints and early stopping
checkpoint_cb = ModelCheckpoint('/home/mustafa/project/save/codementor_model.keras', save_best_only=True)

'''
initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
'''
# Different optimizr from old clean code, hope it works betr than the last one
epochs = 10
initial_learning_rate = 0.1
final_learning_rate = 0.00001
learning_rate_decay_factor = (final_learning_rate / initial_learning_rate)**(1/epochs)
steps_per_epoch = int(len(train_df)/batch_size)

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

#precision_custom = precision_threshold(threshold=0.5, name='precision_custom')
# Initialize the custom precision metric with the desired threshold
# precision_metric = PrecisionThreshold(threshold=0.4)
# Initialize the custom recall metric with the desired threshold
# recall_metric = RecallThreshold(threshold=0.4)

#recall_custom = recall_threshold(threshold=0.5, name='recall_custom')    /// class_weight=class_weight_dict,

model.compile(optimizer=opt, loss=loss_function2,
              metrics=[BinaryAccuracy(),'AUC', 'Precision', 'Recall', 'TruePositives', 'TrueNegatives', 'FalsePositives',
                       'FalseNegatives'])
model.build(input_shape= (128,None,None,None,1))
model.summary()
history = model.fit(training_generator, validation_data=validation_generator, epochs=epochs, verbose=1, workers=10,
                    use_multiprocessing=True,  callbacks=[checkpoint_cb])
