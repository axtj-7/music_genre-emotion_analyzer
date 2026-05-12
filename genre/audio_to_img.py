import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def audio_to_image(audio_path, output_folder, base_name):
    y, sr = librosa.load(audio_path, sr=22050)

    y = librosa.util.normalize(y)
    y, _ = librosa.effects.trim(y)

    segment_length = 10 * sr

    if len(y) < segment_length:
        print(f"Skipped (too short): {audio_path}")
        return

    num_segments = min(5, len(y) // segment_length)

    for i in range(num_segments):
        start = i * segment_length
        end = start + segment_length

        y_segment = y[start:end]

        mel = librosa.feature.melspectrogram(
            y=y_segment,
            sr=sr,
            n_mels=128,
            n_fft=2048,
            hop_length=512
        )

        log_mel = librosa.power_to_db(mel, ref=np.max)

        plt.figure(figsize=(3, 3))
        librosa.display.specshow(log_mel, sr=sr, cmap='magma')
        plt.axis('off')

        output_path = os.path.join(output_folder, f"{base_name}_{i}.png")

        plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()