import cv2
import numpy as np

cap = cv2.VideoCapture('assets/test_video2.mp4')

fps = int(cap.get(cv2.CAP_PROP_FPS))
delay = int(500 / fps)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break 

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_orange = np.array([0, 100, 100])  
    upper_orange = np.array([30, 255, 255])  
    
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 400:
            continue

        epsilon = 0.05 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if 3 <= len(approx) <= 4:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2) 

            cv2.putText(
                frame, 
                "Orange cone", 
                (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 165, 255), 
                2 
            )

    cv2.imshow('Result', frame)

    if cv2.waitKey(delay) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()