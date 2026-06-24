import cv2
import torch
import torch.nn as nn
from torchvision import transforms

# ==========================
# CLASS NAMES
# ==========================

classes = [
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

# ==========================
# CNN MODEL
# ==========================

class GestureCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
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
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ==========================
# LOAD MODEL
# ==========================

device = torch.device("cpu")

model = GestureCNN()

checkpoint = torch.load(
    "models/gesture_cnn.pth",
    map_location=device
)

# Supports both checkpoint formats

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(
        checkpoint
    )

model.eval()

print("Model loaded successfully!")

# ==========================
# TRANSFORM
# ==========================

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# ==========================
# WEBCAM
# ==========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("\nControls:")
print("SPACE -> Save snapshot")
print("Q -> Quit\n")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]

    # ROI SIZE

    size = 250

    x1 = w // 2 - size // 2
    y1 = h // 2 - size // 2

    x2 = x1 + size
    y2 = y1 + size

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        continue

    # ==========================
    # SIMPLE BACKGROUND MASK
    # ==========================

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    _, mask = cv2.threshold(
        gray,
        120,
        255,
        cv2.THRESH_BINARY_INV
    )

    roi_masked = cv2.bitwise_and(
        roi,
        roi,
        mask=mask
    )

    rgb = cv2.cvtColor(
        roi_masked,
        cv2.COLOR_BGR2RGB
    )

    img_tensor = transform(rgb)
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():

        output = model(img_tensor)

        probs = torch.softmax(
            output,
            dim=1
        )

        top3_prob, top3_idx = torch.topk(
            probs,
            3
        )

    # ==========================
    # DRAW ROI BOX
    # ==========================

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # ==========================
    # TOP 3 PREDICTIONS ONLY
    # ==========================

    y_text = 40

    for i in range(3):

        cls = classes[
            top3_idx[0][i].item()
        ]

        prob = (
            top3_prob[0][i].item()
            * 100
        )

        cv2.putText(
            frame,
            f"{i+1}. {cls}: {prob:.1f}%",
            (15, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        y_text += 35

    # ==========================
    # INSTRUCTION
    # ==========================

    cv2.putText(
        frame,
        "Place hand fully inside box",
        (15, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 200, 0),
        2
    )

    cv2.imshow(
        "Hand Gesture Recognition",
        frame
    )

    cv2.imshow(
        "Processed ROI",
        roi_masked
    )

    key = cv2.waitKey(1)

    if key == 32:

        cv2.imwrite(
            "snapshot.jpg",
            frame
        )

        print(
            "Snapshot saved as snapshot.jpg"
        )

    if key & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()