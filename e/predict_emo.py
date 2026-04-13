import os
import sys
import librosa
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from g.audio_feature import extract_audio_features

from collections import Counter

def get_overall_emotion(emotions):
    count = Counter(emotions)
    overall = count.most_common(1)[0][0]
    return overall, count

# Load model + scaler
model = joblib.load("models/emotion_model.pkl")

CHUNK_DURATION = 7  # seconds

def predict_emotion_timeline(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)

    chunk_samples = int(CHUNK_DURATION * sr)
    total_samples = len(y)

    timeline = []
    time_points = []

    for i, start in enumerate(range(0, total_samples, chunk_samples)):
        end = start + chunk_samples

        if end > total_samples:
            break

        chunk = y[start:end]

        feature_dict, feature_vector = extract_audio_features(y=chunk, sr=sr)

        pred = model.predict([feature_vector])[0]

        timeline.append(pred)
        time_points.append(i * CHUNK_DURATION)

    return time_points, timeline


# 🔥 Test
if __name__ == "__main__":
    audio_file = "a.mp3"   # change this

    time_points, emotions = predict_emotion_timeline(audio_file)

    print(list(zip(time_points, emotions)))

    overall_emotion, distribution = get_overall_emotion(emotions)

    print("Overall Emotion:", overall_emotion)
    print("Distribution:", distribution)
    # Plot
    emotion_map = {'sad': 0, 'calm': 1, 'happy': 2, 'angry': 3}
    numeric_emotions = [emotion_map[e] for e in emotions]

    plt.figure(figsize=(10, 3))
    plt.plot(time_points, numeric_emotions, marker='o')

    plt.yticks([0,1,2,3], ['sad','calm','happy','angry'])
    plt.xlabel("Time (seconds)")
    plt.ylabel("Emotion")
    plt.title("Emotion Timeline")
    plt.grid()
    plt.show()