import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
import time as t
import os
from pathlib import Path
import csv
from sklearn.utils.class_weight import compute_class_weight

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, csv_file, base_dir, batch_size=32, shuffle=True):
        """
        Args:
            csv_file: Path to CSV containing 'file_name', 'class', etc.
            base_dir: Directory containing the .npy files
            batch_size: Batch size
            shuffle: Whether to shuffle data after each epoch
        """
        self.data = pd.read_csv(csv_file)
        self.base_dir = Path(base_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_data = self.data.iloc[batch_indexes]

        X = np.empty((len(batch_indexes), 31, 31, 31, 1))
        y = np.empty((len(batch_indexes)), dtype=int)

        for i, (_, row) in enumerate(batch_data.iterrows()):
            print("file name loaded = ", self.base_dir / row['file_name'])
            sample = np.load(self.base_dir / row['file_name'])
            print("sample loaded = ", sample.shape)
            X[i,] = sample.reshape((31, 31, 31, 1))
            y[i] = row['class']

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


class ModelTrainer:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _create_results_dirs(self, model_name):
        """Create directories for model results"""
        model_dir = self.output_dir / f"{model_name}_{self.timestamp}"
        for subdir in ['checkpoints', 'logs', 'metrics']:
            (model_dir / subdir).mkdir(parents=True, exist_ok=True)
        return model_dir

    def _get_metrics(self):
        """Define metrics for model evaluation"""
        return [
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.AUC(curve='PR', name='auc_pr'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.SpecificityAtSensitivity(0.5, name='specificity_at_sensitivity'),
            tf.keras.metrics.FalsePositives(name='false_positives'),
            tf.keras.metrics.FalseNegatives(name='false_negatives'),
            tf.keras.metrics.TruePositives(name='true_positives'),
            tf.keras.metrics.TrueNegatives(name='true_negatives'),
            #BalancedAccuracy(name='balanced_accuracy')  # Your custom metric
        ]

    def _save_metrics(self, history, model_dir, model_name):
        """Save training history and final metrics"""
        # Save training history
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(model_dir / 'logs' / f'{model_name}_training_history.csv', index=False)

        # Save best metrics
        best_metrics = {
            'model_name': model_name,
            'best_epoch': np.argmax(history.history['val_auc']),
            'best_auc_roc': max(history.history['val_auc']),
            'best_auc_pr': max(history.history['val_auc_pr']),
            #'best_balanced_acc': max(history.history['val_balanced_accuracy']),
            'best_sensitivity': max(history.history['val_recall']),
            'best_specificity': max(history.history['val_specificity_at_sensitivity']),
            'best_precision': max(history.history['val_precision']),
            'lowest_val_loss': min(history.history['val_loss']),
            'training_time': self.training_time,
            'timestamp': self.timestamp
        }

        # Save as CSV
        metrics_file = model_dir / 'metrics' / f'{model_name}_best_metrics.csv'
        pd.DataFrame([best_metrics]).to_csv(metrics_file, index=False)

        return best_metrics

    def _get_optimizer(self, model_params):
        """Configure optimizer based on parameters"""
        if model_params['optimizer'] == 'adamw':
            if model_params['lr_schedule'] == 'exponential':
                lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=0.1,
                    decay_steps=1000,
                    decay_rate=0.96
                )
            else:  # cyclic
                lr_schedule = tf.keras.optimizers.schedules.CyclicLR(
                    initial_learning_rate=1e-4,
                    maximal_learning_rate=1e-2,
                    step_size=2000
                )
            return tf.keras.optimizers.AdamW(
                learning_rate=lr_schedule,
                weight_decay=0.004
            )
        else:  # sgd
            if model_params['lr_schedule'] == 'exponential':
                lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=0.1,
                    decay_steps=1000,
                    decay_rate=0.96
                )
            else:  # cyclic
                lr_schedule = tf.keras.optimizers.schedules.CyclicLR(
                    initial_learning_rate=1e-4,
                    maximal_learning_rate=1e-2,
                    step_size=2000
                )
            return tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)

    def _get_class_weights(self, train_data):
        """Calculate class weights from training data"""
        # Get all labels from training data
        all_labels = []
        for _, y in train_data:
            all_labels.extend(y)
        all_labels = np.array(all_labels)

        # Calculate class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(all_labels),
            y=all_labels
        )

        # Convert to dictionary format
        return dict(enumerate(class_weights))

    def _get_loss(self, model_params):
        """Configure loss function based on parameters"""
        if model_params['loss'] == 'focal':
            return tf.keras.losses.BinaryFocalCrossentropy(
                apply_class_balancing=True,
                gamma=2.0
            )
        return tf.keras.losses.BinaryCrossentropy()

    def _create_model(self):
        """Create the CNN model"""
        model = tf.keras.Sequential([
            # Your existing model architecture
            ## define the model's architecture
            tf.keras.layers.Conv3D(filters=16, kernel_size=3, padding='same', input_shape=(31, 31, 31, 1)),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv3D(filters=16, kernel_size=3, padding='same'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool3D(pool_size=2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv3D(filters=32, kernel_size=3, padding='same'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv3D(filters=32, kernel_size=3, padding='same'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool3D(pool_size=2),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv3D(filters=64, kernel_size=3, padding='same'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv3D(filters=64, kernel_size=3, padding='same'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool3D(pool_size=2),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv3D(filters=128, kernel_size=3, padding='same'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv3D(filters=128, kernel_size=3, padding='same'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool3D(pool_size=2),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.GlobalAveragePooling3D(),
            tf.keras.layers.Dense(units=256),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(units=1, activation="sigmoid"),

        ])
        return model

    def train_model(self, model_name, train_data, val_data, model_params):
        """Train a single model configuration"""
        print(f"\nTraining {model_name}")
        model_dir = self._create_results_dirs(model_name)

        # Create callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                str(model_dir / 'checkpoints' / f'{model_name}_best.keras'),  # Changed to .keras
                monitor='val_auc',
                save_best_only=True,
                save_weights_only=False
            ),
            tf.keras.callbacks.CSVLogger(
                str(model_dir / 'logs' / f'{model_name}_training_log.csv')
            ),
            tf.keras.callbacks.TensorBoard(
                str(model_dir / 'logs' / 'tensorboard')
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=10,
                restore_best_weights=True
            )
        ]

        # Create and compile model
        model = self._create_model()
        optimizer = self._get_optimizer(model_params)
        loss = self._get_loss(model_params)

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=self._get_metrics()
        )

        # Calculate class weights if needed
        class_weights = self._get_class_weights(train_data) if model_params['use_class_weights'] else None

        # Train model
        start_time = t.time()
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=50,
            callbacks=callbacks,
            class_weight=class_weights
        )
        self.training_time = t.time() - start_time

        # Save results
        metrics = self._save_metrics(history, model_dir, model_name)

        return model, history, metrics

    def train_model2(self, model_name, train_data, val_data, model_params):
        """Train a single model configuration"""
        print(f"\nTraining {model_name}")
        model_dir = self._create_results_dirs(model_name)

        # Create callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                model_dir / 'checkpoints' / f'{model_name}_best.h5',
                monitor='val_auc',
                save_best_only=True
            ),
            tf.keras.callbacks.CSVLogger(
                model_dir / 'logs' / f'{model_name}_training_log.csv'
            ),
            tf.keras.callbacks.TensorBoard(
                model_dir / 'logs' / 'tensorboard'
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=10,
                restore_best_weights=True
            )
        ]

        # Create and compile model
        model = self._create_model()  # Your existing model architecture

        # Configure optimizer and loss based on model_params
        optimizer = self._get_optimizer(model_params)
        loss = self._get_loss(model_params)

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=self._get_metrics()
        )

        # Train model
        start_time = t.time()
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=50,
            callbacks=callbacks,
            class_weight=self._get_class_weights() if model_params['use_class_weights'] else None
        )
        self.training_time = t.time() - start_time

        # Save results
        metrics = self._save_metrics(history, model_dir, model_name)

        return model, history, metrics


def main():
    # Define paths
    base_dir = Path("/home/mustafa/project/processed_dataset2")
    output_dir = Path("/home/mustafa/project/model_results")

    # Create data generators
    train_data = DataGenerator(
        csv_file=base_dir / "train" / "train_info.csv",
        base_dir=base_dir / "train" / "samples",
        batch_size=32
    )

    val_data = DataGenerator(
        csv_file=base_dir / "val" / "val_info.csv",
        base_dir=base_dir / "val" / "samples",
        batch_size=32,
        shuffle=False
    )

    # Define model configurations
    model_configs = {
        'ADAMW_BCE_ExpDecay': {
            'optimizer': 'adamw',
            'lr_schedule': 'exponential',
            'loss': 'bce',
            'use_class_weights': True
        },
        'ADAMW_BCE_CLR': {
            'optimizer': 'adamw',
            'lr_schedule': 'cyclic',
            'loss': 'bce',
            'use_class_weights': True
        },
        'SGD_BFCE_ExpDecay': {
            'optimizer': 'sgd',
            'lr_schedule': 'exponential',
            'loss': 'focal',
            'use_class_weights': False
        },
        'SGD_BFCE_CLR': {
            'optimizer': 'sgd',
            'lr_schedule': 'cyclic',
            'loss': 'focal',
            'use_class_weights': False
        }
    }

    # Train all models
    trainer = ModelTrainer(output_dir)
    all_results = []

    for model_name, params in model_configs.items():
        model, history, metrics = trainer.train_model(
            model_name,
            train_data,
            val_data,
            params
        )
        all_results.append(metrics)

    # Save comparative results
    comparative_results = pd.DataFrame(all_results)
    comparative_results.to_csv(output_dir / f'comparative_results_{trainer.timestamp}.csv', index=False)


if __name__ == "__main__":
    main()