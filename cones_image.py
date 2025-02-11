import cv2
import numpy as np

image = cv2.imread('assets/test_image3.jpg')

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_orange = np.array([0, 165, 150])  
upper_orange = np.array([15, 255, 255])  

mask = cv2.inRange(hsv, lower_orange, upper_orange)
mask = cv2.erode(mask, None, iterations=1)
mask = cv2.dilate(mask, None, iterations=4)

contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

for cnt in contours:
    
    area = cv2.contourArea(cnt) 
    if area < 300: 
        continue 

    epsilon = 0.05 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    if 3 <= len(approx) <= 4:
        x, y, w, h = cv2.boundingRect(cnt)

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 165, 255), 2) 

        cv2.putText(
                image, 
                "Orange cone", 
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, 
                (0, 165, 255),
                2 
            )

cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()