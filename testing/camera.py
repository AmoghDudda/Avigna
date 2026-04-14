import cv2  
import time
cap=cv2.VideoCapture(0)
#cap=cv2.VideoCapture("esp32 ka ip address")
last_time = time.time()
while True:
    ret,frame=cap.read()
    if not ret:
        break
    cv2.imshow("Avigna",frame)
    if time.time() - last_time >= 1:
        print(f"Shape: {frame.shape}")
        last_time = time.time()
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
