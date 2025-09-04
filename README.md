# Face-Mask-Detection-master
Face Mask Detection project uses deep learning to detect masks in real-time video streams. It leverages MobileNetV2 and OpenCV to classify faces as “with mask” or “without mask.” Includes training and detection scripts with data augmentation for robust performance. Ideal for learning computer vision and AI.
# Face Mask Detection

Real-time face mask detection system using deep learning and computer vision. Utilizes MobileNetV2 for classifying images into "with mask" and "without mask" classes, combined with OpenCV for face detection in live video streams.

## Features

- Real-time mask detection from webcam video
- Based on MobileNetV2 deep learning architecture
- Pre-trained face detection with OpenCV DNN module
- Data augmentation for robust training
- Easy-to-use scripts for training and inference

## Folder Structure

- `dataset/` - Contains images with and without masks for training
- `face_detector/` - Pre-trained face detection model files (deploy.prototxt, caffemodel)
- `detect_mask_video.py` - Runs real-time mask detection on video stream
- `train_mask_detector.py` - Script to train mask detection model
- `mask_detector.h5` - Saved Keras model file
- `requirements.txt` - Python dependencies
- `plot.png` - Training accuracy & loss graph

## Installation

1. Clone the repository: https://github.com/Rahulpedimalla/Face-Mask-Detection-master

2. Install dependencies: pip install -r requirements.txt

## Usage

### Training the Model

Run the training script to train or fine-tune the model: python train_mask_detector.py

### Running Mask Detection

Start the webcam mask detection: python detect_mask_video.py

## Requirements

- Python 3.6 or above
- TensorFlow and Keras
- OpenCV
- imutils, numpy, matplotlib, scipy

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MobileNetV2 model by Google
- OpenCV DNN face detector
- Dataset contributors

---

*Created as a practical project for learning deep learning and computer vision techniques.*
