
# AI Hand Gesture Recognition

A real-time hand gesture recognition system built using Python, OpenCV, MediaPipe, and Machine Learning.

## Features
- Detects hand landmarks using MediaPipe.
- Recognizes five hand gestures:
  - Open Palm
  - Fist
  - Thumbs Up
  - Peace Sign
  - Rock Sign
- Displays prediction confidence and FPS.

## Technologies Used
- Python
- OpenCV
- MediaPipe
- Scikit-learn
- Pandas
- NumPy

## How It Works
1. Detects 21 hand landmarks using MediaPipe.
2. Extracts 63 landmark coordinates as features.
3. Trains a Random Forest classifier.
4. Recognizes gestures in real time using a webcam.

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the prediction program:

```bash
python predict.py
```

Press `q` to exit the webcam window.

## Author

Mohammed Halith  
B.Tech Artificial Intelligence and Data Science
