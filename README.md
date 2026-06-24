# 👍Hand Gesture Recognition using CNN

A real-time Hand Gesture Recognition System built using **PyTorch**, **OpenCV**, and **Convolutional Neural Networks (CNNs)**. The model is trained on the LeapGestRecog dataset and can recognize multiple hand gestures through a webcam for intuitive human-computer interaction.

## 📌 Features

* Real-time hand gesture recognition using webcam
* Custom CNN architecture built with PyTorch
* Trained on the LeapGestRecog dataset
* Supports 10 different hand gestures
* Displays Top-3 predictions with confidence scores
* Live webcam inference using OpenCV
* Model checkpoint saving and loading
* Stratified train-test split for balanced evaluation

---

## 📂 Dataset

Dataset used:

**LeapGestRecog Dataset**

Kaggle: https://www.kaggle.com/datasets/gti-upm/leapgestrecog

The dataset contains 20,000 images distributed equally among 10 gesture classes.

### Supported Gestures

| Label | Gesture    |
| ----- | ---------- |
| 0     | Palm       |
| 1     | L          |
| 2     | Fist       |
| 3     | Fist Moved |
| 4     | Thumb      |
| 5     | Index      |
| 6     | OK         |
| 7     | Palm Moved |
| 8     | C          |
| 9     | Down       |

---

## 🏗️ Project Structure

```text
HandGestureRecognition/
│
├── dataset/
│   └── leapGestRecog/
│
├── models/
│   └── gesture_cnn.pth
│
├── src/
│   ├── train_model.py
│   └── webcam_predict.py
│
├── requirements.txt
│
└── README.md
```

---

## ⚙️ Technologies Used

* Python
* PyTorch
* TorchVision
* OpenCV
* NumPy
* Pillow
* Scikit-Learn

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/your-username/HandGestureRecognition.git

cd HandGestureRecognition
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Model Training

Run:

```bash
python src/train_model.py
```

The trained model will automatically be saved in:

```text
models/gesture_cnn.pth
```

---

## 🎥 Real-Time Gesture Prediction

Run:

```bash
python src/webcam_predict.py
```

### Controls

| Key   | Action           |
| ----- | ---------------- |
| SPACE | Save Snapshot    |
| Q     | Quit Application |

The webcam window displays:

* Live video feed
* Gesture recognition predictions
* Top-3 gesture probabilities
* Confidence scores

---

## 📊 Model Architecture

The CNN consists of:

* 4 Convolutional Layers
* Batch Normalization
* ReLU Activation
* Max Pooling
* Adaptive Average Pooling
* Fully Connected Layers
* Dropout Regularization



---

## 📈 Results

### Dataset Statistics

* Total Images: 20,000
* Classes: 10
* Training Images: 16,000
* Testing Images: 4,000

### Performance

* Training Accuracy: ~99.9%
* Test Accuracy: ~99–100%

*Actual webcam performance may vary depending on lighting conditions, camera quality, background, and hand positioning.*

---

## Demo Video
 https://drive.google.com/file/d/1YAZjlZlKvQUy6XqecXWdzQLWVMXPigWK/view?usp=drive_link

## 🔮 Future Improvements

* MediaPipe Hand Detection Integration
* Transfer Learning using ResNet18
* Dynamic Gesture Recognition
* Sign Language Recognition
* GPU Training Support
* Deployment using Flask/Streamlit

---

## 👨‍💻 Author

**Nishitha Pallati**

Machine Learning Internship project using PyTorch and OpenCV.
