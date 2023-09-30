import os
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.utils import Sequence

# Constants
MODEL_PATH = "3d_image_classification.h5"
BASE_DIR = "/home/mustafa/project/LUNA16/cropped_nodules/"
CSV_PATH = "/home/mustafa/project/LUNA16/cropped_nodules.csv"

# Load Model
def load_saved_model(model_path):
    model = keras.models.load_model(model_path)
    print(f"Loaded model from disk: {model_path}")
    model.summary()
    return model

def preprocess_data(csv_path, base_dir):
    nodules_csv = pd.read_csv(csv_path)
    all_image_paths = sorted(os.listdir(base_dir), key=lambda x: int(os.path.splitext(x)[0]))
    all_image_paths = np.array(all_image_paths)
    nodules = nodules_csv.rename(columns = {'SN':'ID'})

    abnormal_nodules = nodules.loc[nodules['state'] == 1]
    normal_nodules = nodules.loc[nodules['state'] == 0]

    train_data, train_labels = create_training_data(abnormal_nodules, normal_nodules, all_image_paths)
    val_data, val_labels = create_validation_data(nodules, train_data, all_image_paths)

    return train_data, train_labels, val_data, val_labels


def create_training_data(abnormal_nodules, normal_nodules, all_image_paths):
    # select 90% of data for training
    train_nodules = pd.concat([abnormal_nodules.sample(frac=0.90), normal_nodules.sample(frac=0.90)])

    train_ids = train_nodules['ID'].tolist()
    train_labels = train_nodules['state'].tolist()

    train_ids_str = map(str, train_ids)
    train_data = [item + '.npy' for item in train_ids_str]

    return train_data, train_labels


def create_validation_data(nodules, train_data, all_image_paths):
    filtered_array = np.isin(all_image_paths, train_data, invert=True)
    val_data = all_image_paths[filtered_array]

    val_labels_df = nodules.loc[~nodules['ID'].isin(train_data)]
    val_labels = val_labels_df['state'].tolist()

    return val_data, val_labels


class DataGenerator(Sequence):
    def __init__(self, all_image_paths, labels, base_dir, output_size, shuffle=False, batch_size=10):
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
        X = np.empty((self.batch_size, *self.output_size, 1))
        y = np.empty((self.batch_size, 1))

        indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        for i, data_index in enumerate(indices):
            img_path = os.path.join(self.base_dir, self.imgs[data_index])
            img = np.load(img_path)

            while img.shape == (31, 31, 31):
                img = np.expand_dims(img, axis=3)

                label = self.labels[data_index]

                X[i,] = img
                y[i] = label

        return X, y


# Main execution
if __name__ == "__main__":
    model = load_saved_model(MODEL_PATH)
    train_data, train_labels, val_data, val_labels = preprocess_data(CSV_PATH, BASE_DIR)
    train_gen = DataGenerator(train_data, train_labels, BASE_DIR, (31, 31, 31), batch_size=128, shuffle=True)
    val_gen = DataGenerator(val_data, val_labels, BASE_DIR, (31, 31, 31), batch_size=128, shuffle=True)

    checkpoint_cb = keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True)
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_precision", patience=10, mode='max',
                                                      restore_best_weights=True)

    history = model.fit(train_gen,
        validation_data=val_gen,
        epochs=1,
        shuffle=True,
        verbose=1,
        callbacks=[checkpoint_cb, early_stopping_cb],
    )

