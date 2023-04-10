import cv2
import numpy as np

# Load the image in grayscale
img = cv2.imread('night_sky.png', cv2.IMREAD_GRAYSCALE)

# Apply a threshold to the image to separate the stars from the background
thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]

# Find the contours of the bright areas in the thresholded image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter the contours based on their size and shape
min_contour_area = 1  # Change this to adjust the minimum size of the contours
max_contour_sides = 25  # Change this to adjust the maximum number of sides for the contours
roundness_threshold = 0.3  # Change this to adjust the threshold for roundness (between 0 and 1)
stars = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > min_contour_area:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) <= max_contour_sides and cv2.isContourConvex(approx):
            perimeter = cv2.arcLength(cnt, True)
            roundness = (4 * np.pi * area) / (perimeter * perimeter)
            if roundness >= roundness_threshold:
                stars.append(cnt)

# Draw the contours of the stars on a copy of the original image
img_with_stars = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(img_with_stars, stars, -1, (0, 0, 255), 2)

# Count the number of remaining contours
num_stars = len(stars)

# Print the number of stars and show the image with the stars highlighted
print(f"Number of round stars: {num_stars}")
cv2.imshow('Image with round stars', img_with_stars)
cv2.waitKey(0)
cv2.destroyAllWindows()
