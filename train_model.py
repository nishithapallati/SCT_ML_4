import os
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit

# =====================
# SETTINGS
# =====================

DATASET_PATH = "dataset/leapGestRecog"
MODEL_PATH = "models/gesture_cnn.pth"

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# =====================
# LABELS
# =====================

gesture_map = {
    "01_palm": 0,
    "02_l": 1,
    "03_fist": 2,
    "04_fist_moved": 3,
    "05_thumb": 4,
    "06_index": 5,
    "07_ok": 6,
    "08_palm_moved": 7,
    "09_c": 8,
    "10_down": 9
}

class_names = [
    "Palm",
    "L",
    "Fist",
    "FistMoved",
    "Thumb",
    "Index",
    "OK",
    "PalmMoved",
    "C",
    "Down"
]

# =====================
# DATASET
# =====================

class GestureDataset(Dataset):

    def __init__(self, root_dir):
        self.samples = []

        for subject in sorted(os.listdir(root_dir)):

            subject_path = os.path.join(root_dir, subject)

            if not os.path.isdir(subject_path):
                continue

            for gesture in sorted(os.listdir(subject_path)):

                if gesture not in gesture_map:
                    continue

                gesture_path = os.path.join(subject_path, gesture)

                label = gesture_map[gesture]

                for img in os.listdir(gesture_path):

                    if img.lower().endswith(
                        (".jpg", ".jpeg", ".png")
                    ):
                        self.samples.append(
                            (
                                os.path.join(
                                    gesture_path,
                                    img
                                ),
                                label
                            )
                        )

        print(
            f"Total samples loaded: {len(self.samples)}"
        )

        counts = Counter(
            [label for _, label in self.samples]
        )

        for idx, name in enumerate(class_names):
            print(
                f"{name:<12}: {counts[idx]} images"
            )

    def __len__(self):
        return len(self.samples)

    def get_labels(self):
        return [label for _, label in self.samples]


full_dataset = GestureDataset(DATASET_PATH)

# =====================
# TRANSFORMS
# =====================

train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1)
    ),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# =====================
# SPLIT DATA
# =====================

labels = full_dataset.get_labels()

splitter = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(
        range(len(full_dataset)),
        labels
    )
)

class GestureSubset(Dataset):

    def __init__(
        self,
        dataset,
        indices,
        transform
    ):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        path, label = self.dataset.samples[
            self.indices[idx]
        ]

        image = Image.open(path).convert("RGB")

        image = self.transform(image)

        return image, label


train_dataset = GestureSubset(
    full_dataset,
    train_idx,
    train_transform
)

test_dataset = GestureSubset(
    full_dataset,
    test_idx,
    test_transform
)

print(f"\nTrain size: {len(train_dataset)}")
print(f"Test size : {len(test_dataset)}")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# =====================
# MODEL
# =====================

class GestureCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3, 32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                128, 256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256 * 4 * 4,
                512
            ),

            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(
                512,
                10
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


model = GestureCNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    patience=3,
    factor=0.5
)

# =====================
# TRAINING
# =====================

best_acc = 0.0

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    train_acc = (
        100 * correct / total
    )

    # Validation

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

    test_acc = (
        100 * val_correct / val_total
    )

    scheduler.step(test_acc)

    lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch+1:02d}/{EPOCHS} | "
        f"Loss={running_loss:.2f} | "
        f"Train={train_acc:.2f}% | "
        f"Test={test_acc:.2f}% | "
        f"LR={lr:.6f}"
    )

    if test_acc > best_acc:

        best_acc = test_acc

        os.makedirs(
            "models",
            exist_ok=True
        )

        torch.save(
            {
                "model_state_dict":
                model.state_dict(),
                "accuracy":
                test_acc
            },
            MODEL_PATH
        )

        print(
            f"✓ Best model saved "
            f"({test_acc:.2f}%)"
        )

print(
    f"\nTraining Complete!"
)

print(
    f"Best Accuracy: {best_acc:.2f}%"
)