import os
import sys
import pandas as pd
import numpy as np
import librosa
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -----------------------------------
# FIX IMPORT PATH
# -----------------------------------
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..'
        )
    )
)

from g.audio_feature import extract_audio_features

# -----------------------------------
# LOAD CSV
# -----------------------------------
df = pd.read_csv(
    "../dataset/emotion/emotion_labels.csv"
)

X = []
y = []

CHUNK_DURATION = 7  # seconds

# -----------------------------------
# FEATURE EXTRACTION WITH SEGMENTATION
# -----------------------------------
for row in df.itertuples():
    print(f"Processing: {row.filename}")
    file = row.filename
    label = row.label

    path = os.path.join(
        "../dataset/emotion",
        label,
        file
    )

    if not os.path.exists(path):

        print(f"File not found: {path}")
        continue

    try:

        y_audio, sr = librosa.load(
            path,
            sr=22050
        )

        chunk_samples = int(
            CHUNK_DURATION * sr
        )

        total_samples = len(y_audio)

        for start in range(
            0,
            total_samples,
            chunk_samples
        ):

            end = start + chunk_samples

            # Skip incomplete last chunk
            if end > total_samples:
                break

            chunk = y_audio[start:end]

            feature_dict, feature_vector = extract_audio_features(
                y=chunk,
                sr=sr
            )
            
            X.append(feature_vector)
            y.append(label)

    except Exception as e:

        print(f"Error processing {path}: {e}")

# -----------------------------------
# DATASET
# -----------------------------------
X = np.array(X)
y = np.array(y)

print(
    "\nTotal samples after segmentation:",
    len(X)
)

print(
    "Class distribution:",
    {
        label: list(y).count(label)
        for label in set(y)
    }
)

# -----------------------------------
# TRAIN TEST SPLIT
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)

# -----------------------------------
# RANDOM FOREST MODEL
# -----------------------------------
model = RandomForestClassifier(

    n_estimators=300,

    max_depth=20,

    max_features="sqrt",

    min_samples_split=5,

    min_samples_leaf=2,

    random_state=42,

    class_weight="balanced",

    n_jobs=-1
)

# -----------------------------------
# TRAINING
# -----------------------------------
print("\n🚀 Training Random Forest...\n")

model.fit(X_train, y_train)

# -----------------------------------
# PREDICTION
# -----------------------------------
pred = model.predict(X_test)

# -----------------------------------
# ACCURACY
# -----------------------------------
acc = accuracy_score(
    y_test,
    pred
)

print(f"\n✅ Emotion Accuracy: {acc:.4f}")

# -----------------------------------
# CLASSIFICATION REPORT
# -----------------------------------
print("\n📄 Classification Report:\n")

print(classification_report(
    y_test,
    pred
))

# -----------------------------------
# CONFUSION MATRIX
# -----------------------------------
cm = confusion_matrix(
    y_test,
    pred
)

labels = sorted(list(set(y)))

plt.figure(figsize=(6, 5))

sns.heatmap(

    cm,

    annot=True,

    fmt='d',

    cmap='Purples',

    xticklabels=labels,

    yticklabels=labels
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title(
    "Emotion Recognition Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "emotion_confusion_matrix.png"
)

plt.show()

print(
    "\n🖼️ Confusion matrix saved as "
    "emotion_confusion_matrix.png"
)

# -----------------------------------
# SAVE MODEL
# -----------------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/emotion_model.pkl"
)

print(
    "\n✅ Model saved as "
    "models/emotion_model.pkl"
)