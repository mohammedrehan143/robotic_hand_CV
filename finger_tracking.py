import cv2
import mediapipe as mp

# -------- MediaPipe Setup -------- #
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# -------- Camera Setup -------- #
cap = cv2.VideoCapture(0)

# Optional resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

margin = 15  # Stability threshold

while True:
    success, frame = cap.read()
    if not success:
        print("❌ Camera error")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    detected_hand = "None"
    total_fingers = 0

    thumb_open = index_open = middle_open = ring_open = pinky_open = 0

    if results.multi_hand_landmarks and results.multi_handedness:

        for handLms, handedness in zip(results.multi_hand_landmarks,
                                       results.multi_handedness):

            label = handedness.classification[0].label
            detected_hand = label

            h, w, c = frame.shape
            landmarks = []

            # -------- Draw Landmarks -------- #
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append((cx, cy))

                # Dot
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), cv2.FILLED)

                # Landmark number
                cv2.putText(frame, str(id),
                            (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0, 0, 255),
                            1)

            # Skeleton
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            # -------- Finger Detection -------- #

            # 👍 Thumb (fixed left/right)
            if label == "Right":
                thumb_open = 0 if landmarks[4][0] < landmarks[3][0] - margin else 1
            else:
                thumb_open = 0 if landmarks[4][0] > landmarks[3][0] + margin else 1

            # ✋ Other fingers
            index_open  = 1 if landmarks[8][1]  < landmarks[6][1]  - margin else 0
            middle_open = 1 if landmarks[12][1] < landmarks[10][1] - margin else 0
            ring_open   = 1 if landmarks[16][1] < landmarks[14][1] - margin else 0
            pinky_open  = 1 if landmarks[20][1] < landmarks[18][1] - margin else 0

            total_fingers = thumb_open + index_open + middle_open + ring_open + pinky_open

    # -------- Stationary UI -------- #

    cv2.putText(frame, f'Hand Detected: {detected_hand}',
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2)

    cv2.putText(frame, f'Fingers Open: {total_fingers}',
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2)

    # -------- Individual Finger States -------- #

    cv2.putText(frame, f"Thumb: {'Open' if thumb_open else 'Closed '}", (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"Index: {'Open' if index_open else 'Closed'}", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"Middle: {'Open' if middle_open else 'Closed'}", (20, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"Ring: {'Open' if ring_open else 'Closed'}", (20, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"Pinky: {'Open' if pinky_open else 'Closed'}", (20, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Hand Finger States", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
