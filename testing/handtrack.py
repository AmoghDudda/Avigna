import cv2
import time
import mediapipe as mp
import numpy as np
import math
 
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)


def fingers_extended(landmarks):
    fingers = [] 
    if landmarks[4].x < landmarks[3].x:
        fingers.append(True)
    else:
        fingers.append(False)
    for tip,pip in [(8,6), (12,10), (16,14), (20,18)]:       
        if landmarks[tip].y < landmarks[pip].y:
            fingers.append(True)
        else:
            fingers.append(False)
    return fingers

def pinch_distance(landmarks):
    x1,y1 = landmarks[4].x, landmarks[4].y
    x2,y2 = landmarks[8].x, landmarks[8].y
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def classify_gesture(landmarks):
    f = fingers_extended(landmarks)
    # f[0]=thumb f[1]=index f[2]=middle f[3]=ring f[4]=pinky
    
    dist = pinch_distance(landmarks)
    
    if dist < 0.05 and not f[2] and not f[3] and not f[4]:
        return "PINCH"
    elif f == [False, False, False, False, False]:
        return "FIST"
    elif f == [True, True, True, True, True]:
        return "OPEN PALM"
    elif f[1] and not f[2] and not f[3] and not f[4]:
        return "POINT"
    elif f[1] and f[2] and not f[3] and not f[4]:
        return "TWO FINGER"
    elif f[0] and not f[1] and not f[2] and not f[3] and not f[4]:
        return "THUMBS UP"
    elif not f[0] and not f[1] and f[2] and not f[3] and not f[4]:
        return"fkoff"
    else:
        return "UNKNOWN"



cap = cv2.VideoCapture(1)



while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results   = hands.process(frame_rgb)

    gesture = "NO HAND"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks,
                                   mp_hands.HAND_CONNECTIONS)
            gesture = classify_gesture(hand_landmarks.landmark)
    if gesture == "OPEN PALM":
        colour = (0, 255, 0)
    elif gesture == "FIST":
        colour = (0, 0, 255)
    elif gesture == "POINT":
        colour = (255, 0, 0)
    elif gesture == "PINCH":
        colour = (0, 255, 255)
    elif gesture == "TWO FINGER":
        colour = (255, 255, 0)
    else:
        colour = (255, 255, 255)

    cv2.putText(frame, gesture, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 2)

    cv2.imshow("Avigna", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()