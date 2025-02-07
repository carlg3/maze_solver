import cv2, numpy as np

filename = 'maze.png'
img = cv2.imread(filename)

# Binary conversion
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Inverting tholdolding will give us a binary image with a white wall and a black background.
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# Contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

dc = cv2.drawContours(thresh, contours, 0, (255, 255, 255), 5)
dc = cv2.drawContours(dc, contours, 1, (0, 0, 0), 5)

ret, thresh = cv2.threshold(dc, 240, 255, cv2.THRESH_BINARY)

kernel = np.ones((19, 19), np.uint8)

# Dilate
dilation = cv2.dilate(thresh, kernel, iterations = 1)

# Erosion
erosion = cv2.erode(dilation, kernel, iterations = 1)

# Find differences between two images
diff = cv2.absdiff(dilation, erosion)

mask_inv = cv2.bitwise_not(diff)

# Splitting the channels of maze
b, g, r = cv2.split(img)

# Masking out the green and red colour from the solved path
r = cv2.bitwise_and(r, r, mask=mask_inv)
b = cv2.bitwise_and(b, b, mask=mask_inv)

res = cv2.merge((b, g, r))
cv2.imwrite(filename+'_solved.png', res)