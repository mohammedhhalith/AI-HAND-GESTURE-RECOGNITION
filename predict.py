
import cv2
import mediapipe as mp
import pickle
import time

# Load trained ML model
with open("gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize webcam
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# FPS variables
prev_time = 0

while True:

    success, frame = cap.read()

    if not success:
        print("Camera error")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame, cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    # Default display
    gesture = "No Hand"
    confidence = 0.0

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Extract 63 features
            row = []

            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])

            # Predict gesture
            prediction = model.predict([row])[0]

            # Get prediction probabilities
            probabilities = model.predict_proba([row])[0]

            # Get confidence of predicted gesture
            confidence = max(probabilities) * 100

            gesture = prediction

    # Calculate FPS
    current_time = time.time()

    if prev_time != 0:
        fps = 1 / (current_time - prev_time)
    else:
        fps = 0

    prev_time = current_time

    # Display gesture
    cv2.putText(
        frame,
        "Gesture: " + gesture,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Display confidence
    cv2.putText(
        frame,
        "Confidence: " + str(round(confidence, 2)) + "%",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    # Display FPS
    cv2.putText(
        frame,
        "FPS: " + str(round(fps, 1)),
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    # Show webcam
    cv2.imshow(
        "AI Gesture Recognition - Stage 4",
        frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
hands.close()
cv2.destroyAllWindows()