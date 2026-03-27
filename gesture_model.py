"""
TensorFlow gesture classifier for Rock-Paper-Scissors.
Uses a small Dense neural network trained on hand landmark features extracted by MediaPipe.
"""

import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF warnings

import tensorflow as tf
from tensorflow import keras


MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "rps_model.h5")
GESTURES = ["rock", "paper", "scissors"]


def _build_model():
    """Build a small Dense network for gesture classification."""
    model = keras.Sequential([
        keras.layers.Input(shape=(63,)),  # 21 landmarks × 3 coords (x, y, z)
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(3, activation="softmax"),  # rock, paper, scissors
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def generate_synthetic_data(samples_per_class=500):
    """Generate synthetic training data based on typical hand landmark patterns.
    
    This creates approximate landmark distributions for rock/paper/scissors gestures
    based on the characteristic finger positions for each gesture.
    
    Returns:
        X: numpy array of shape (N, 63)
        y: numpy array of shape (N,) with labels 0=rock, 1=paper, 2=scissors
    """
    np.random.seed(42)
    X_all = []
    y_all = []

    for _ in range(samples_per_class):
        # ROCK: All fingers curled in (tips close to palm, low y-values relative to MCP)
        landmarks = np.random.normal(0.5, 0.05, (21, 3))
        # Curl all fingertips toward palm
        for tip_id in [4, 8, 12, 16, 20]:
            landmarks[tip_id, 1] = landmarks[tip_id - 2, 1] + np.random.normal(0.05, 0.02)
            landmarks[tip_id, 0] = landmarks[tip_id - 2, 0] + np.random.normal(0.0, 0.02)
        # Keep fingers close together
        for i in range(21):
            landmarks[i, 0] += np.random.normal(0, 0.01)
        X_all.append(landmarks.flatten())
        y_all.append(0)

    for _ in range(samples_per_class):
        # PAPER: All fingers extended (tips far from wrist, spread apart)
        landmarks = np.random.normal(0.5, 0.05, (21, 3))
        # Extend all fingertips (lower y values = higher on screen = extended)
        for i, tip_id in enumerate([4, 8, 12, 16, 20]):
            landmarks[tip_id, 1] = landmarks[tip_id - 2, 1] - np.random.normal(0.08, 0.02)
            landmarks[tip_id, 0] = 0.3 + i * 0.1 + np.random.normal(0, 0.02)
        X_all.append(landmarks.flatten())
        y_all.append(1)

    for _ in range(samples_per_class):
        # SCISSORS: Index and middle extended, others curled
        landmarks = np.random.normal(0.5, 0.05, (21, 3))
        # Extend index and middle
        for tip_id in [8, 12]:
            landmarks[tip_id, 1] = landmarks[tip_id - 2, 1] - np.random.normal(0.08, 0.02)
        # Curl ring, pinky, thumb
        for tip_id in [4, 16, 20]:
            landmarks[tip_id, 1] = landmarks[tip_id - 2, 1] + np.random.normal(0.05, 0.02)
        # Spread index and middle apart
        landmarks[8, 0] -= np.random.normal(0.03, 0.01)
        landmarks[12, 0] += np.random.normal(0.03, 0.01)
        X_all.append(landmarks.flatten())
        y_all.append(2)

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int32)

    # Shuffle
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def train_model(epochs=50, save=True):
    """Train the RPS gesture model on synthetic data.
    
    Args:
        epochs: Number of training epochs
        save: Whether to save the model to disk
        
    Returns:
        Trained Keras model
    """
    print("Generating synthetic training data...")
    X, y = generate_synthetic_data(samples_per_class=1000)

    # Split into train/validation
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples...")
    model = _build_model()
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        verbose=1,
    )

    if save:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        model.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

    return model


class GestureClassifier:
    """Classifies hand landmarks into rock/paper/scissors gestures."""

    def __init__(self):
        self.model = None
        self._load_or_train()

    def _load_or_train(self):
        """Load pre-trained model or train a new one."""
        if os.path.exists(MODEL_PATH):
            print("Loading gesture model...")
            self.model = keras.models.load_model(MODEL_PATH)
        else:
            print("No pre-trained model found. Training new model...")
            self.model = train_model()

    def predict(self, landmark_array):
        """Predict gesture from hand landmarks.
        
        Args:
            landmark_array: numpy array of shape (63,) from HandTracker.get_landmark_array()
            
        Returns:
            tuple: (gesture_name, confidence) where gesture_name is 'rock', 'paper', or 'scissors'
        """
        if landmark_array is None:
            return None, 0.0

        input_data = landmark_array.reshape(1, -1)
        prediction = self.model.predict(input_data, verbose=0)
        class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][class_idx])

        return GESTURES[class_idx], confidence

    def predict_from_tracker(self, tracker):
        """Predict gesture directly from a HandTracker instance.
        
        Args:
            tracker: HandTracker instance that has already processed a frame
            
        Returns:
            tuple: (gesture_name, confidence)
        """
        landmarks = tracker.get_landmark_array()
        return self.predict(landmarks)


if __name__ == "__main__":
    print("Training Rock-Paper-Scissors gesture model...")
    model = train_model(epochs=50)
    print("Done! Model saved.")
