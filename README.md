# Real-Time Indian Sign Language (ISL) Word-to-Text Recognition

A deep learning project that recognizes Indian Sign Language (ISL) words from images and live webcam video, converting them into real-time sequential text.

> **Note: This system is trained and works on 6 Indian Sign Language words only.**

---

## Target Vocabulary
1. **AGREE**
2. **DOCTOR**
3. **PAIN**
4. **PRAY**
5. **WORK**
6. **YOU**

---

## Project Structure

```text
isl_word_detection/
├── app/
│   └── app.py               # Real-time webcam word-to-text recognition application
├── dataset/                 # Dataset organized by word and user folders
│   ├── agree/
│   ├── doctor/
│   ├── pain/
│   ├── pray/
│   ├── work/
│   └── you/
├── models/                  # Model weights
│   └── best_model.pth       # Trained checkpoint
├── results/                 # Evaluation figures and training plots
│   ├── audit_contact_sheet.png
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── recognized_output.txt
├── src/                     # Core source code
│   ├── audit_data.py        # Dataset audit and integrity verification
│   ├── config.py            # Central configurations, hyperparameters, and paths
│   ├── dataset.py           # Dataset loaders and augmentations
│   ├── evaluate.py          # Model evaluation on test/validation set
│   ├── hand_crop.py         # MediaPipe hand detection module
│   ├── model.py             # Model architecture definition
│   └── train.py             # Model training pipeline
├── .gitignore               # Ignores virtual environment and cached files
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Audit Dataset
Inspect folder structure, file integrity, and participant distribution:
```bash
python src/audit_data.py
```

### 3. Train Model
Train the deep learning model with transfer learning:
```bash
python src/train.py
```

### 4. Evaluate Model
Evaluate performance and generate the confusion matrix:
```bash
python src/evaluate.py
```

### 5. Run Real-Time Webcam Application
Launch the live webcam interface:
```bash
python app/app.py
```

#### Webcam Controls:
- **`C`** : Clear recognized text stream
- **`D`** or **`Backspace`** : Delete the last recognized word
- **`K`** : Save current recognized text to file
- **`Q`** or **`ESC`** : Exit application cleanly
