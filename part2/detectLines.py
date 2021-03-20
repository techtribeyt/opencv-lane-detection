import cv2
import numpy as np

def show_image(title, img):
    cv2.imshow(title, img)
    cv2.waitKey()
    cv2.destroyAllWindows()

SIZE = 500
img = cv2.imread("road_2.jpg")
img = cv2.resize(img, (SIZE, SIZE))

show_image("Original", img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect edges
edges = cv2.Canny(gray, 150, 300)

# create mask
mask = np.zeros(img.shape[:2], dtype = "uint8") # 0 - 255 = 8 bits

# white pentagon
pts = np.array([[0, SIZE * 3 / 4], [SIZE / 2, SIZE / 2], [SIZE, SIZE * 3 / 4], [SIZE, SIZE], [0, SIZE]], np.int32)

# black triangle
pts2 = np.array([[SIZE / 2, 0], [SIZE / 4, SIZE], [SIZE * 3 / 4, SIZE]], np.int32)

cv2.fillPoly(mask, [pts], 255)

cv2.fillPoly(mask, [pts2], 0)

show_image("mask", mask)

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
        
line_img = cv2.bitwise_and(line_img, line_img, mask = mask)

overlay = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)

show_image("Overlay", overlay)
