
import cv2
import mediapipe as mp
import csv
import os

# Open webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Drawing utility
mp_draw = mp.solutions.drawing_utils

# Create CSV file
file_name = "gesture_data.csv"

file_exists = os.path.isfile(file_name)

f = open(file_name, "a", newline="")
writer = csv.writer(f)

# Add header if file is new
if not file_exists:
    header = []

    for i in range(21):
        header.append(f"x{i}")
        header.append(f"y{i}")
        header.append(f"z{i}")

    header.append("label")
    writer.writerow(header)

print("Press O = Open Palm")
print("Press F = Fist")
print("Press T = Thumbs Up")
print("Press P = Peace Sign")
print("Press Q = Quit")

label = ""

while True:

    success, frame = cap.read()

    if not success:
        print("Camera error")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Collect landmark coordinates
            if label != "":

                row = []

                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])

                row.append(label)

                writer.writerow(row)

                print("Saved:", label)

    cv2.putText(
        frame,
        "Label: " + label,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Gesture Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('o'):
        label = "Open Palm"

    elif key == ord('f'):
        label = "Fist"

    elif key == ord('t'):
        label = "Thumbs Up"

    elif key == ord('p'):
        label = "Peace Sign"

    elif key == ord('r'):
        label = "Rock Sign"

    elif key == ord('q'):
        break

    elif key == ord('x'):
        label = ""

# Release resources
f.close()
cap.release()
hands.close()
cv2.destroyAllWindows()
