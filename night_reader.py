import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image in grayscale
img = cv2.imread('night_sky.png', cv2.IMREAD_GRAYSCALE)
liczba = []
# Apply a threshold to the image to separate the stars from the background
for i in range(250):
    thresh = cv2.threshold(img, i, 255, cv2.THRESH_BINARY)[1]

    # Find the contours of the bright areas in the thresholded image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter the contours based on their size
    min_contour_area = 10  # Change this to adjust the minimum size of the contours
    stars = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Count the number of remaining contours
    num_stars = len(stars)
    liczba.append(num_stars)

    print(f"Number of stars: {num_stars}")

print(liczba)
numbers = list(map(lambda x: x, range(1, 251)))

p = np.polyfit(numbers, liczba, 12)

# Generate x values for plotting
x = np.linspace(numbers[0], numbers[-1], 100)

# Evaluate the polynomial at each x value
y = np.polyval(p, x)

# Plot the data and the curve
plt.plot(numbers, liczba, 'o', label='Data')
plt.plot(x, y, label='Fit')
plt.legend()
plt.show()



plt.show()