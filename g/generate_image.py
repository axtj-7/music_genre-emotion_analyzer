import os
from audio_to_img import audio_to_image

DATASET_PATH = "dataset"
OUTPUT_PATH = "images"

# Create main output folder
os.makedirs(OUTPUT_PATH, exist_ok=True)

for genre in os.listdir(DATASET_PATH):
    genre_path = os.path.join(DATASET_PATH, genre)

    # Skip if not a folder
    if not os.path.isdir(genre_path):
        continue

    output_genre_path = os.path.join(OUTPUT_PATH, genre)
    os.makedirs(output_genre_path, exist_ok=True)

    for file in os.listdir(genre_path):
        if file.lower().endswith((".wav", ".au", ".flac", ".mp3")):

            audio_path = os.path.join(genre_path, file)
            base_name = os.path.splitext(file)[0]

            try:
                # 🔥 Correct call (IMPORTANT)
                audio_to_image(audio_path, output_genre_path, base_name)
                print(f"✅ Done: {file}")

            except Exception as e:
                print(f"❌ Skipped {file}: {e}")