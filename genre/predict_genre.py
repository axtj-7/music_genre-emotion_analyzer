import torch
from genre.model_architecture import CNN
from torchvision import transforms
from PIL import Image
import os
from collections import Counter
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from audio_to_img import audio_to_image

# -----------------------------
# CLASS NAMES (must match training folders)
# -----------------------------
class_names = ['classical', 'hiphop', 'jazz', 'pop', 'rock']
num_classes = len(class_names)


# -----------------------------
# LOAD MODEL
# -----------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = CNN(num_classes).to(device)
model.load_state_dict(torch.load("genre/models/model.pth", map_location=device))
model.eval()

# -----------------------------
# IMAGE TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


# -----------------------------
# PREDICT FUNCTION
# -----------------------------
def predict(audio_path):
    # Generate spectrogram segments
    audio_to_image(audio_path, ".", "temp")

    predictions = []
    confidences = []

    # Loop through all temp segment images
    for file in os.listdir("."):
        if file.startswith("temp_") and file.endswith(".png"):

            img = Image.open(file).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)

            predictions.append(class_names[predicted.item()])
            confidences.append(confidence.item())

            os.remove(file)  # cleanup

    # Majority voting
    final_label = Counter(predictions).most_common(1)[0][0]
    final_conf = sum(confidences) / len(confidences)

    return final_label, final_conf


# -----------------------------
# TEST
# -----------------------------
if __name__ == "__main__":
    test_file = "dataset/test/a.mp3"

    label, conf = predict(test_file)

    print("\n🎵 Prediction:", label)
    print("🔥 Confidence:", round(conf, 3))