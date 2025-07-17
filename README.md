# SCT_ML_04

# 🤖 Hand Gesture Recognition using CNN

This project implements a **Convolutional Neural Network (CNN)** to recognize static hand gestures from the **LeapGestRecog** dataset. It includes data loading, preprocessing, model training, evaluation, and test image prediction — all using Python, OpenCV, TensorFlow, and Matplotlib.

---

## 📂 Dataset

**LeapGestRecog** from Kaggle:  .(https://www.kaggle.com/datasets/gti-upm/leapgestrecog).
Each gesture class is organized in folders under user IDs (e.g., `01`, `02`, etc.).

/leapGestRecog/
├── 01/
│ ├── 01_palm/
│ │ ├── img_1.png
│ │ ├── ...
│ └── ...
└── ...


> ✅ Only one user folder (`01`) is used for simplicity. You can expand to more users.

---

## 🚀 Workflow

1. 📁 Load grayscale gesture images
2. 🔖 Label gestures using folder names
3. ⚙️ Normalize and reshape input
4. 🧠 Train CNN on image data
5. 📊 Evaluate test accuracy
6. 🔍 Predict and visualize user-provided image

---

## 🧠 Model Architecture

```python
Sequential([
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
```
pip install tensorflow opencv-python matplotlib numpy
