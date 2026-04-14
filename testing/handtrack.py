import cv2
import time
import mediapipe as mp
import numpy as np
 
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)
last_time = time.time()
while True:
    ret,frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                lm=hand_landmarks.landmark[8]
                h,w,_ = frame.shape
                cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 15, (0,255,0), -1)
            if time.time() - last_time >= 1:    
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    lm=hand_landmarks.landmark[8]
                    h,w,_ = frame.shape
                    print(f"Label:{results.multi_handedness[i].classification[0].label}")
                    # print(f"x={lm.x:.2f} y={lm.y:.2f} z={lm.z:.2f} ")
                    px = int(lm.x*w)
                    py = int(lm.y*h)
                    print(f"pixel x={px} pixel y={py}")
                    
                    

                last_time = time.time()
                
    cv2.imshow("Avigna Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

