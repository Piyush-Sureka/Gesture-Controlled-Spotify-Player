##### **Gesture-Controlled Spotify Media Controller**

A real-time hand gesture recognition system that enables touch-free Spotify media control using MediaPipe, PyTorch, and a Temporal Convolutional Network (TCN).

###### **Overview**

This project captures dynamic hand gestures through a webcam, extracts dynamic 3D hand landmarks using MediaPipe Hands, and classifies gesture sequences using a Temporal Convolutional Network (TCN). The predicted gesture is translated into Spotify media commands such as Play/Pause, Next Track, Previous Track, and Volume Control.

###### **Features**

* Real-time hand tracking using MediaPipe Hands
* Temporal Convolutional Network (TCN) for gesture sequence classification
* Spotify media playback control
* Live webcam overlay displaying gesture and confidence
* 5-Fold Stratified Cross Validation evaluation
* Separate deployment training pipeline (train_final.py)
* Lightweight deployment model (~1.07 MB)

###### **Supported Gestures**

* Swipe Left (Represented by a Right Hand L)
* Swipe Right (Represented by a Left Hand L)
* Circle (Represented by a fist)
* Wave (Represented by a open palm)
* None / Idle

###### **Model Performance**

Evaluation Method - 5-Fold Stratified Cross Validation

Performance - Accuracy: 98.40% - Precision: 98.46% - Recall: 98.40% - F1-Score: 98.38%

Model - Parameters: 279,557 - Model Size: 1.07 MB

###### **System Pipeline**

Webcam → MediaPipe Hands → 21 Hand Landmarks (63 Features) → Landmark Normalization → 30-Frame Temporal Buffer → Temporal Convolutional Network (TCN) → Gesture Prediction → Spotify Media Commands

###### **Repository Structure**


Gesture-Controlled-Spotify-Player/
+---assets/
|       architecture.png
|       confusion_matrix.png
|       demo.gif
|       gestures.png
|       overlay.png
|       
+---data/...
|       
+---synthetic_data/...
|   
|   .env.example
|   .gitignore
|   capture.py
|   evaluation_report.txt
|   generate_synthetic.py
|   inference.py
|   inference_spotify_overlay.py
|   README.md
|   requirements.txt
|   tcn_gesture.pth
|   tcn_model.py
|   train.py
|   train_final.py
|   visualize_gesture.py

###### **Installation**

pip install -r requirements.txt

###### **Spotify API Configuration**

Before running the application, create a Spotify Developer application and set the following environment variables.

1. Create a Spotify application from the Spotify Developer Dashboard.
2. Copy `.env.example` to `.env`.
3. Replace the placeholder values with your own Spotify API credentials.

You can obtain the Client ID and Client Secret by creating an application in the Spotify Developer Dashboard.

###### **Training**

Evaluation: python train.py –classes swipe_left swipe_right circle wave none

Deployment: python train_final.py –classes swipe_left swipe_right circle wave none

###### **Inference**

python inference_spotify_overlay.py –model tcn_gesture.pth –classes swipe_left swipe_right circle wave none

###### **Future Improvements**

* Increase dataset size

* Add more gesture classes

* Improve lighting robustness

* Support additional media players

* Deploy as a desktop application

###### **Author**

Piyush Sureka
GitHub: https://github.com/Piyush-Sureka