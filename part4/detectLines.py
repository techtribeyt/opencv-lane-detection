import cv2
import numpy as np

def show_image(title, img):
    cv2.imshow(title, img)
    cv2.waitKey()
    cv2.destroyAllWindows()

SIZE = 500
img = cv2.imread("road_2.jpg")
img = cv2.resize(img, (SIZE, SIZE))
original_img = img.copy()

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

# get average line
def get_length_threshold(lines):
    lengths = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            lengths.append(((x2-x1) ** 2 + (y2-y1) ** 2) ** 0.5)
    
    # set threshold to top 80% longest lines
    return np.quantile(lengths, 0.8)


left_counter = right_counter = 0
left_x, left_y, right_x, right_y = [], [], [], []
length_threshold = get_length_threshold(lines)



for line in lines:
    for x1, y1, x2, y2 in line:
        # for every line
        
        if x1 == x2: continue # to avoid division by 0
        
        # in code, y is positive down
        slope = (y1 - y2) / (x2 - x1)
        length = ((x2-x1) ** 2 + (y2-y1) ** 2) ** 0.5
        
        # ensure only long lines are considered
        if length < length_threshold: continue
                
        # these coords belong to right line
        if slope < 0:
            right_counter += 1
            right_x += [x1, x2]
            right_y += [y1, y2]
        # left line
        else:
           left_counter += 1
           left_x += [x1, x2]
           left_y += [y1, y2] 
        
        
# calculate linear fit
BOTTOM_Y = img.shape[0]
TOP_Y = img.shape[0] * 3 // 5
LANE_COLOR = (0, 255, 0)

def draw_average_line(x_list, y_list, counter):
    if counter > 0:
        polyfit = np.polyfit(y_list, x_list, deg = 1)
        poly = np.poly1d(polyfit)
        x_start = int(poly(BOTTOM_Y))
        x_end = int(poly(TOP_Y))
        cv2.line(img, (x_start, BOTTOM_Y), (x_end, TOP_Y), LANE_COLOR, 5)
        return (x_start, x_end)

(left_start, left_end) = draw_average_line(left_x, left_y, left_counter)
(right_start, right_end) = draw_average_line(right_x, right_y, right_counter)

show_image("Image", img)

final_image = original_img.copy()
MIN_Y = SIZE * 2 // 3
MAX_Y = SIZE


# lane drawing
center_lane_x_start = (right_start + left_start) // 2
center_lane_x_end = (right_end + left_end) // 2 
cv2.line(final_image, (center_lane_x_start, MAX_Y), (center_lane_x_end, MIN_Y), LANE_COLOR, 15)

# car trajectory drawing
x_car = SIZE // 2
CAR_CENTER_COLOR = (255, 0, 0)
cv2.line(final_image, (x_car, MAX_Y), (x_car, MIN_Y), CAR_CENTER_COLOR, 5)


# show final image
show_image("Final", final_image)







