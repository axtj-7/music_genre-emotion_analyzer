import os
import csv

base_path = "dataset/emotion"
output_csv = "dataset/emotion/emotion_labels.csv"

rows = []

for label in os.listdir(base_path):
    folder_path = os.path.join(base_path, label)

    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith((".wav", ".mp3")):
                rows.append([file, label])

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label"])
    writer.writerows(rows)

print("CSV created successfully!")