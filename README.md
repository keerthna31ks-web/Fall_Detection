# 🚨 AI-Based Fall Detection System

An AI-powered video analysis system that detects human falls using **Deep Learning (ResNet18 + LSTM)** and provides prediction confidence through an interactive Streamlit dashboard.

# AI-Based Fall Detection System

🚀 **Live Demo:** [Click here to try the application](https://falldetection-pb9vmm487szr5zaety6uvq.streamlit.app/)


## 📌 Project Overview

Falls can cause serious injuries, especially among elderly people. This project aims to detect potential fall events from video clips using a deep learning model.

The system analyzes video frames, extracts spatial features using ResNet18, and learns temporal patterns using LSTM to classify activities as **Normal** or **Fall**.

## ✨ Features

- 🎥 Upload videos for fall detection
- 🧠 Deep learning model using ResNet18 + LSTM
- 🔍 Analyzes sequences of 32 video frames
- 📊 Displays prediction confidence
- 📈 Visualizes Normal and Fall probabilities
- 🚨 Safety alert when a possible fall is detected
- 🖥️ Interactive dashboard built with Streamlit

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| PyTorch | Deep learning framework |
| ResNet18 | Spatial feature extraction |
| LSTM | Temporal sequence learning |
| OpenCV | Video frame processing |
| Streamlit | Interactive web application |
| NumPy | Numerical operations |

## 🏗️ System Architecture

```text
        Input Video
             ↓
      Video Processing
             ↓
     Extract 32 Frames
             ↓
      Resize to 224×224
             ↓
       ResNet18 CNN
  (Spatial Feature Extraction)
             ↓
          LSTM
 (Temporal Feature Learning)
             ↓
      Fully Connected Layer
             ↓
    ┌────────┴────────┐
    ↓                 ↓
  Normal             Fall
    ↓                 ↓
 Confidence       Safety Alert
    └────────┬────────┘
             ↓
    Streamlit Dashboard
```

## 📂 Project Structure

```text
Fall_Detection/
│
├── models/
│   └── best_32frame.pth
│
├── videos/
│   ├── normal.mp4
│   └── fall.mp4
│
├── model.py
├── video_processor.py
├── app.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Fall_Detection.git
cd Fall_Detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows:**

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🧪 How to Use

1. Launch the Streamlit application.
2. Upload a video containing normal activity or a potential fall.
3. Click **Analyze Video**.
4. Wait for the model to process the video.
5. View the predicted activity and confidence scores.
6. If a fall is detected, a safety alert will be displayed.

## 📊 Model Information

| Component | Description |
|---|---|
| CNN | ResNet18 |
| Sequence Model | LSTM |
| Input Frames | 32 |
| Frame Resolution | 224 × 224 |
| Output Classes | Normal, Fall |
| Framework | PyTorch |

## ⚠️ Disclaimer

This project is developed for **educational and research purposes**. It is a prototype and is not a certified medical or emergency monitoring system. Predictions may be inaccurate. Do not rely on this system as the sole basis for real-world emergency decisions.

## 👩‍💻 Author

**Keerthana S**  
B.Tech – Artificial Intelligence and Data Science

---

⭐ If you find this project interesting, consider giving the repository a star!
