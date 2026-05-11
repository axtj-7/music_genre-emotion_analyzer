import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# SETTINGS
# -----------------------------
DATASET_PATH = "images"
BATCH_SIZE = 16
IMG_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.001

# -----------------------------
# CREATE MODELS DIRECTORY
# -----------------------------
os.makedirs("models", exist_ok=True)

# -----------------------------
# TRANSFORMS + AUGMENTATION
# -----------------------------
transform = transforms.Compose([

    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )
])

# -----------------------------
# DATASET
# -----------------------------
dataset = datasets.ImageFolder(
    DATASET_PATH,
    transform=transform
)

print("Classes:", dataset.classes)

# -----------------------------
# TRAIN / VALIDATION SPLIT
# -----------------------------
train_idx, val_idx = train_test_split(
    list(range(len(dataset))),
    test_size=0.2,
    stratify=dataset.targets,
    random_state=42
)

train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

num_classes = len(dataset.classes)

# -----------------------------
# MODEL
# -----------------------------
class CNN(nn.Module):

    def __init__(self, num_classes):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128, 256),
            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        x = self.conv(x)
        x = self.pool(x)
        x = self.fc(x)

        return x

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print(f"\nUsing Device: {device}")

model = CNN(num_classes).to(device)

# -----------------------------
# LOSS + OPTIMIZER
# -----------------------------
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# -----------------------------
# LEARNING RATE SCHEDULER
# -----------------------------
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=3
)

# -----------------------------
# TRAINING
# -----------------------------
print("\n🚀 Training Started...\n")

best_acc = 0

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        correct += (predicted == labels).sum().item()

        total += labels.size(0)

    train_accuracy = 100 * correct / total

    # -----------------------------
    # VALIDATION
    # -----------------------------
    model.eval()

    val_correct = 0
    val_total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            val_correct += (predicted == labels).sum().item()

            val_total += labels.size(0)

            all_preds.extend(
                predicted.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    val_accuracy = 100 * val_correct / val_total

    scheduler.step(val_accuracy)

    print(
        f"Epoch {epoch+1}/{EPOCHS} "
        f"| Loss: {total_loss:.3f} "
        f"| Train Accuracy: {train_accuracy:.2f}% "
        f"| Val Accuracy: {val_accuracy:.2f}%"
    )

    # -----------------------------
    # SAVE BEST MODEL
    # -----------------------------
    if val_accuracy > best_acc:

        best_acc = val_accuracy

        torch.save(
            model.state_dict(),
            "models/model.pth"
        )

        print("✅ Best model saved!")

# -----------------------------
# FINAL EVALUATION
# -----------------------------
print("\n📊 FINAL MODEL EVALUATION\n")

print(f"✅ Best Validation Accuracy: {best_acc:.2f}%")

# -----------------------------
# CLASSIFICATION REPORT
# -----------------------------
print("\n📄 Classification Report:\n")

print(classification_report(
    all_labels,
    all_preds,
    target_names=dataset.classes
))

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(
    all_labels,
    all_preds
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=dataset.classes,
    yticklabels=dataset.classes
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title(
    "Genre Classification Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "genre_confusion_matrix.png"
)

plt.show()

print(
    "\n🖼️ Confusion matrix saved as "
    "genre_confusion_matrix.png"
)

print(
    "\n✅ Final best model saved as "
    "models/model.pth"
)