# Building a sign language recognition model using the Resnet-50 network
# Introduction
This project creates a model using the ResNet-50 architecture to recognize Vietnamese sign language, including 22 unaccented letters.
# Features
•	Trained ResNet-50 model for sign language classification using TensorFlow.

•	Processed real-time image frames from webcam input using OpenCV. 

•	Used MediaPipe library for hand detection and cropped hand images to feed into prediction model. 

•	Integrated entire pipeline for real-time sign language recognition and prediction.

# Data preparation
<img width="895" height="964" alt="image" src="https://github.com/user-attachments/assets/d2b3832d-81b2-42b6-9cc9-a8e963e177ca" />

https://drive.google.com/file/d/1YWv20_vgqOFXh-kURzb5nuD3RxYuG6fG/view?usp=sharing

The dataset was constructed by capturing hand gesture images, which were then preprocessed to ensure consistency in format, scale, and labeling. The final dataset comprises 22 folders, each representing a distinct Vietnamese sign language symbol, with 2,000 images per folder. To support model training and evaluation, the dataset was randomly split into two subsets: 80% for training and 20% for testing.

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

When the camera is activated, the user makes a Vietnamese sign language gesture. The system recognizes the hand and crops the hand image, processes it with a trained recognition model, displays the recognized letter on the interface and simultaneously plays the corresponding sound.

# Results
https://www.youtube.com/watch?v=epobEQixPjo&list=PLP9z0n_UuqoEGXa90ebJ_4UI__IXJwimX
# Advantages
• High accuracy in good lighting conditions and when users perform hand signs clearly and correctly.

• Intuitive user interface, supporting users to operate conveniently and track recognition results clearly.

• Pronunciation function helps increase interactivity.

• The system supports both recognition modes: by single letters and by continuous character strings, suitable for many different purposes.
# Limitations
• Accuracy degrades in low-light conditions and complex backgrounds, or when the hand is partially obscured, making it difficult for the system to extract full features.

• Some pairs of symbols have similar shapes (such as “I” and “Y”,) leading to confusion during recognition.

• The system has a delay when starting up, and sometimes the recognition speed is slow due to the impact of the device's processing performance.

• The system only recognizes static symbols.
