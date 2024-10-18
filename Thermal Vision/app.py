import cv2
import numpy as np

# Load the nighttime traffic image
image = cv2.imread('E:/technical/Projects/Traffic Management/Thermal Vision/night_traffic.jpg')

# Convert the image to grayscale (CLAHE works better on grayscale images)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(gray)

# Apply Gaussian Blur to reduce noise
blurred_image = cv2.GaussianBlur(clahe_image, (5, 5), 0)

# Convert CLAHE result back to BGR format and combine with the original image
clahe_bgr = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR)
combined_image = cv2.addWeighted(clahe_bgr, 0.6, image, 0.4, 0)

# Save and display the final enhanced image
cv2.imwrite('enhanced_traffic_image.jpg', combined_image)
cv2.imshow('Enhanced Night Traffic Image', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
