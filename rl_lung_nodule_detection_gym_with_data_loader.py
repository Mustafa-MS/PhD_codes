import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from sklearn.utils import class_weight
import gym
from gym import spaces

class NoduleDataGenerator(Sequence):
    # ... (same as in data_loader.py)
    def __init__(self, image_paths, labels, batch_size, input_shape, shuffle=True):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        batch_image_paths = self.image_paths[start:end]
        batch_labels = self.labels[start:end]

        X = np.empty((self.batch_size, *self.input_shape))
        y = np.empty((self.batch_size, 1))

        for i, image_path in enumerate(batch_image_paths):
            X[i,] = np.load(image_path)
            y[i,] = batch_labels[i]

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            zipped = list(zip(self.image_paths, self.labels))
            random.shuffle(zipped)
            self.image_paths, self.labels = zip(*zipped)


def load_data(image_dir, truth_table_path, test_size=0.2, validation_size=0.1):
    # ... (same as in data_loader.py)
    truth_table = pd.read_csv(truth_table_path)
    image_paths = [os.path.join(image_dir, f"{i}.npy") for i in truth_table['ImageIndex']]
    labels = truth_table['Label'].values

    X_train, X_test_val, y_train, y_test_val = train_test_split(image_paths, labels, test_size=test_size, stratify=labels, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=validation_size / (1 - test_size), stratify=y_test_val, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test


class LungNoduleGymEnvironment(gym.Env):
    # ... (same as in rl_lung_nodule_detection_gym.py)
    def __init__(self, image_path, model, n_classes):
        super().__init__()
        self.image = np.load(image_path)
        self.model = model
        self.n_classes = n_classes
        self.window_size = (31, 31, 31)
        self.position = [0, 0, 0]

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=0, high=255, shape=self.window_size, dtype=np.uint8)

    def step(self, action):
        if action == 0:  # Move up
            self.position[1] = min(self.position[1] + 1, self.image.shape[1] - self.window_size[1])
        elif action == 1:  # Move down
            self.position[1] = max(self.position[1] - 1, 0)
        elif action == 2:  # Move right
            self.position[0] = min(self.position[0] + 1, self.image.shape[0] - self.window_size[0])
        elif action == 3:  # Move left
            self.position[0] = max(self.position[0] - 1, 0)
        elif action == 4:  # Move front
            self.position[2] = min(self.position[2] + 1, self.image.shape[2] - self.window_size[2])
        elif action == 5:  # Move back
            self.position[2] = max(self.position[2] - 1, 0)

        window = self.image[self.position[0]:self.position[0] + self.window_size[0],
                            self.position[1]:self.position[1] + self.window_size[1],
                            self.position[2]:self.position[2] + self.window_size[2]]

        window_normalized = window[np.newaxis, :] / 255.0
        predictions = self.model.predict(window_normalized)
        class_label = np.argmax(predictions)
        confidence = predictions[0, class_label]

        reward = confidence if class_label != 0 else -confidence
        done = True if confidence > 0.9 else False
        info = {'position': self.position, 'class_label': class_label, 'confidence': confidence}

        return window, reward, done, info

    def reset(self):
        self.position = [0, 0, 0]
        return self.image[self.position[0]:self.position[0] + self.window_size[0],
                          self.position[1]:self.position[1] + self.window_size[1],
                          self.position[2]:self.position[2] + self.window_size[2]]

def build_and_train_cnn(train_generator, val_generator, input_shape, num_classes):
    # ... (same as in rl_lung_nodule_detection_gym.py)
    model = Sequential()
    model.add(Conv3D(32, (3, 3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv3D(64, (3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test), class_weight=class_weights)

    return model


def rl_agent(env, model, episodes, max_steps, epsilon, epsilon_decay, gamma):
    # ... (same as in rl_lung_nodule_detection_gym.py)
    q_table = np.zeros((env.action_space.n,))

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                window_normalized = state[np.newaxis, :] / 255.0
                predictions = model.predict(window_normalized)
                action = np.argmax(predictions)

            next_state, reward, done, info = env.step(action)
            total_reward += reward

            q_table[action] = q_table[action] + gamma * (reward + np.max(q_table) - q_table[action])

            state = next_state

            if done:
                print(f'Episode {episode + 1}: Detected nodule with label {info["class_label"]} and confidence {info["confidence"]:.2f}')
                break

        epsilon *= epsilon_decay

    return q_table


def main():
    # Load and preprocess data
    image_dir = 'path/to/nodule/images'
    truth_table_path = 'path/to/truth_table.csv'
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(image_dir, truth_table_path)

    input_shape = (31, 31, 31)
    num_classes = 2
    batch_size = 32

    train_generator = NoduleDataGenerator(X_train, y_train, batch_size, input_shape)
    val_generator = NoduleDataGenerator(X_val, y_val, batch_size, input_shape)
    test_generator = NoduleDataGenerator(X_test, y_test, batch_size, input_shape)

    # Build and train the CNN model
    model = build_and_train_cnn(train_generator, val_generator, input_shape, num_classes)

    # Reinforcement learning parameters
    episodes = 100
    max_steps = 1000
    epsilon = 1
    epsilon_decay = 0.99
    gamma = 0.9

    # Create the OpenAI Gym environment
    image_path = os.path.join('x', 'image.npy')
    env = LungNoduleGymEnvironment(image_path, model, num_classes)

    # Run the RL agent
    rl_agent(env, model, episodes, max_steps, epsilon, epsilon_decay, gamma)

if __name__ == '__main__':
    main()
