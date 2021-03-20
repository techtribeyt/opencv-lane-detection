import cv2
import numpy as np

img = cv2.imread("road.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect edges
edges = cv2.Canny(gray, 150, 300)


# get lines
# (x1, y1, x2, y2)
lines = cv2.HoughLinesP(
    edges,
    rho=1.0,
    theta=np.pi/180,
    threshold=20,
    minLineLength=30,
    maxLineGap=10        
)

# draw lines
line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
line_color = [0, 255, 0]
line_thickness = 2
dot_color = [0, 255, 0]
dot_size = 3

for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(line_img, (x1, y1), (x2, y2), line_color, line_thickness)
        cv2.circle(line_img, (x1, y1), dot_size, dot_color, -1)
        cv2.circle(line_img, (x2, y2), dot_size, dot_color, -1)

overlay = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
cv2.imshow("Overlay", overlay)
cv2.waitKey()
cv2.destroyAllWindows()
