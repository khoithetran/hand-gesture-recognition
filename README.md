# ✋ Hand Gesture Recognition with Mediapipe and TensorFlow

This project builds a real-time hand gesture recognition system using keypoints extracted from webcam video and a custom-trained neural network classifier.

## 🎯 Objective

- Detect hand landmarks in real-time using MediaPipe Hands  
- Collect labeled keypoint data for various gestures  
- Train a neural network classifier using TensorFlow  
- Predict gestures live through webcam input  

## 📦 Dataset

The gesture dataset is custom-built by:

- Capturing hand keypoints via webcam using MediaPipe  
- Saving each gesture class (`thumbs_up`, `okay`, `peace`, etc.) as `.npy` files  
- Each sample includes 21 landmarks (`x`, `y`, `z`), resulting in a feature vector of shape (63,)  

You can extend the dataset by adding your own gestures and retraining the model.

## 🤖 Model

We use a simple fully connected neural network implemented in TensorFlow/Keras:

- **Input:** 63 features (21 hand keypoints × 3)  
- **Hidden layers:** Dense + ReLU  
- **Output:** Softmax layer for gesture classification  

The final model is exported as `gesture_classifier.h5`.

## 🧪 Usage

Run real-time gesture prediction with:

```bash
python GUI.py
