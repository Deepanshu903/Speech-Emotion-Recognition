# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1YaI7T22QQPKYZ_jFsPiIJ714ijR1kGpC
"""

# test_model.py

import sys
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Constants
SAMPLE_RATE = 22050
DURATION = 3
N_MELS = 128
EXPECTED_SHAPE = (128, 128)
EMOTIONS = ['neutral','calm','happy','sad','angry','fear','disgust','surprise']

# === Load Saved Artifacts ===
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

class Attention(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="random_normal", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)
model = load_model('emotion_model.h5', compile=False, custom_objects={'Attention': Attention})
mean = np.load('train_mean.npy')
std = np.load('train_std.npy')

# Audio Feature Extraction
def extract_log_mel(path):
    y, _ = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION)
    if len(y) < SAMPLE_RATE * DURATION:
        y = np.pad(y, (0, SAMPLE_RATE * DURATION - len(y)), mode='constant')
    else:
        y = y[:SAMPLE_RATE * DURATION]

    S = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS)
    S_db = librosa.power_to_db(S, ref=np.max)

    if S_db.shape[1] < 128:
        S_db = np.pad(S_db, ((0,0), (0, 128 - S_db.shape[1])), mode='constant')
    else:
        S_db = S_db[:, :128]

    return S_db

# Prediction Function
def predict_emotion(file_path):
    mel = extract_log_mel(file_path)
    mel = (mel - mean) / std
    mel = mel[np.newaxis, ..., np.newaxis]
    pred = model.predict(mel)
    pred_class = np.argmax(pred)
    print(f"\nPredicted Emotion: {EMOTIONS[pred_class]} (Confidence: {pred[0][pred_class]:.2f})")

# CLI Entry
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_model.py <audio_file.wav>")
        sys.exit(1)
    audio_file = sys.argv[1]
    predict_emotion(audio_file)