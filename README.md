# Building a sign language recognition model using the Resnet-50 network
# Introduction
This project creates a model using the ResNet-50 architecture to recognize Vietnamese sign language, including 22 unaccented letters.
# Features
•	Trained ResNet-50 model for sign language classification using TensorFlow. 
•	Processed real-time image frames from webcam input using OpenCV. 
•	Used MediaPipe library for hand detection and cropped hand images to feed into prediction model. 
•	Integrated entire pipeline for real-time sign language recognition and prediction.
# Training model architecture
<img width="422" height="640" alt="Picture1" src="https://github.com/user-attachments/assets/917edb64-0f72-4c5d-b2c7-aec5e4eec7a0" />
The original fully connected layers of the ResNet-50 architecture have been removed and replaced with a custom classification head tailored for Vietnamese sign language recognition. This modification allows the model to effectively learn and differentiate between 22 distinct sign language gestures, improving its performance on this specialized task.
<img width="832" height="317" alt="image" src="https://github.com/user-attachments/assets/794f8f61-8963-460e-94ab-2ab49a7c4ad4" />
<img width="703" height="583" alt="image" src="https://github.com/user-attachments/assets/d39e8643-5cfe-44fe-98f0-7e0bb21833f9" />

The model was trained for 20 epochs using a batch size of 32 and a learning rate of 0.00001.
# Display interface
<img width="601" height="338" alt="image" src="https://github.com/user-attachments/assets/e6f230ef-030e-4ccc-ae7e-c5ebd4e779d0" />
The user interface is designed for intuitive interaction: functional buttons are placed on the left for easy access, the central area displays the live camera feed, and recognition results are shown in a dedicated bar beneath the video stream.
# Identification process
<img width="536" height="198" alt="image" src="https://github.com/user-attachments/assets/4f68ace2-628c-4aae-a8b3-99883749a903" />
When the camera is activated, the user performs a Vietnamese sign language gesture. The system will crop the image of the hand, process it with a trained recognition model, display the recognized letter on the interface and simultaneously play the corresponding sound.

