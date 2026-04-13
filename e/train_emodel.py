import os
import sys
import pandas as pd
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from g.audio_feature import extract_audio_features

# Load CSV
df = pd.read_csv("dataset/emotion/emotion_labels.csv")

X = []
y = []

CHUNK_DURATION = 7  # seconds

# Extract features with segmentation
for row in df.itertuples():
    file = row.filename
    label = row.label

    # ✅ Correct path using label
    path = os.path.join("dataset/emotion", label, file)

    if not os.path.exists(path):
        print(f"File not found: {path}")
        continue

    try:
        y_audio, sr = librosa.load(path, sr=22050)

        chunk_samples = int(CHUNK_DURATION * sr)
        total_samples = len(y_audio)

        for start in range(0, total_samples, chunk_samples):
            end = start + chunk_samples

            # Skip last small chunk
            if end > total_samples:
                break

            chunk = y_audio[start:end]

            # Extract features directly from chunk
            features = extract_audio_features(y=chunk, sr=sr)

            X.append(features)
            y.append(label)

    except Exception as e:
        print(f"Error processing {path}: {e}")

X = np.array(X)
y = np.array(y)

print("Total samples after segmentation:", len(X))
print("Class distribution:", {label: list(y).count(label) for label in set(y)})


# Train-test split (with stratification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

print("Emotion Accuracy:", acc)

# Save model and scaler
joblib.dump(model, "models/emotion_model.pkl")

print("Model saved!")