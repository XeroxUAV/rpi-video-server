import cv2

cap = cv2.VideoCapture("rtsp://<PI_IP>:8554/live")
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Drone RTSP", frame)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
