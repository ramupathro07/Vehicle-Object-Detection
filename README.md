# Vehicle Object Detection using YOLOv8

A custom-trained **YOLOv8** object detection model for detecting vehicles (cars, trucks, etc.) in images and videos. The model was trained on a custom dataset from **Roboflow Universe**.

![Demo](https://github.com/ramupathro07/Vehicle-Object-Detection/raw/main/results/demo.gif) <!-- Add your output video/gif here -->

## 🚀 Project Overview

This project demonstrates how to train a state-of-the-art YOLOv8 model on a custom vehicle detection dataset using **Ultralytics YOLOv8** and **Roboflow**.

### Features
- Custom vehicle detection (Cars, Trucks, etc.)
- Trained on Roboflow dataset
- Google Colab notebook for easy training
- Inference on images and videos
- High accuracy with YOLOv8 architecture

## 📁 Dataset
- **Source**: [Roboflow Universe - Vehicles Dataset](https://universe.roboflow.com/eddie-cotter/vehicles-5ej3n)
- **Format**: YOLOv8
- **Classes**: Cars, Trucks, and other vehicles
- **Images**: 9,711+ annotated images

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/ramupathro07/Vehicle-Object-Detection.git
cd Vehicle-Object-Detection
```
## 2. Open in Google Colab

<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">

## 📋 Step-by-Step Training Process

### Step 1: Environment Setup

Open the notebook in Google Colab
Go to Runtime → Change runtime type → Hardware accelerator → GPU (T4)
Run the first cells to check GPU and install Ultralytics:
```
!nvidia-smi
!pip install ultralytics==8.2.103 -q
```

### Step 2: Download Dataset from Roboflow
```
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("eddie-cotter").project("vehicles-5ej3n")
version = project.version(1)
dataset = version.download("yolov8")
```

### Step 3: Train the Model

```
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s.pt")  # or yolov8n.pt, yolov8m.pt etc.

# Train the model
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=25,           # You used 10, recommended 25-50
    imgsz=800,
    plots=True
)
```
### Step 4: Run Inference on Video
```
# Inference on video
results = model.predict(
    source="path/to/your/input_video.mp4",
    save=True,
    conf=0.4
)
```
The output video will be saved in runs/detect/predict/ folder.

## 📊 Results

- **Best Model:** runs/detect/train/weights/best.pt
- Trained for 25 epochs (you can adjust)
- Supports real-time detection on videos

## 📁 Project Structure

```
Vehicle-Object-Detection/
├── notebooks/
│   └── train-yolov8-object-detection-on-custom-dataset.ipynb
├── data/
├── runs/                     # Training results & predictions
├── results/                  # Demo videos & images
├── README.md
└── requirements.txt
```

## 🎯 How to Use the Trained Model

- Download best.pt from the runs/detect/train/weights/ folder
- Use the following code for inference:

```
from ultralytics import YOLO

model = YOLO("path/to/best.pt")

# Predict on video
model.predict(source="input_video.mp4", save=True, conf=0.4)
```

## 🔧 Technologies Used

- YOLOv8 (Ultralytics)
- Roboflow (Dataset management)
-  Google Colab (GPU Training)
- Python 3.12
- PyTorch + CUDA

## 📝 Future Improvements

- Increase epochs for better accuracy
- Add more classes (bikes, buses, etc.)
- Deploy using Roboflow Inference or Flask API
- Real-time webcam detection

## 🤝 Acknowledgments

- Ultralytics YOLOv8
- Roboflow
- Eddie Cotter (Dataset creator)

### Made with ❤️ by Ramu
#### If you find this project helpful, please Star ⭐ the repository!


