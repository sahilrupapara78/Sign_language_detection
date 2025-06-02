# Sign Language Detection System

A real-time sign language recognition system using LSTM neural networks and MediaPipe hand tracking to detect and classify ASL (American Sign Language) alphabets.

## Features
✔ **Real-time Detection**: Recognizes hand signs from live camera feed  
✔ **26 ASL Letters**: Supports A-Z alphabet detection  
✔ **Deep Learning Model**: LSTM neural network for sequence recognition  
✔ **MediaPipe Integration**: Accurate hand landmark detection  
✔ **Training Pipeline**: Complete workflow from data collection to model deployment  

## Technologies Used
- **Computer Vision**: OpenCV, MediaPipe
- **Deep Learning**: Keras, TensorFlow
- **Model Architecture**: LSTM Neural Network
- **Data Processing**: NumPy, Pandas

## System Architecture
1. **Data Collection**: `collectdata.py` captures hand images for each letter
2. **Feature Extraction**: `data.py` processes images into landmark sequences
3. **Model Training**: `trainmodel.py` trains the LSTM classifier
4. **Real-time Detection**: `app.py` runs the live prediction system

## Installation
```bash
git clone https://github.com/yourusername/sign-language-detection.git
cd sign-language-detection
pip install -r requirements.txt
```

## Requirements
- Python 3.8+  
- OpenCV (pip install opencv-python)  
- MediaPipe (pip install mediapipe)  
- TensorFlow (pip install tensorflow)  
- NumPy (pip install numpy)
