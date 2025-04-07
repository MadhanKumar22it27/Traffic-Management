import cv2

# Load rainy image
img = cv2.imread('input.png')


# Increase median filter strength
blurred = cv2.medianBlur(img, 5)  # from 3 to 5

# Use bilateral filter as alternative to guided filter
guided = cv2.bilateralFilter(blurred, d=9, sigmaColor=75, sigmaSpace=75)


cv2.imwrite('cleaned_image.jpg', guided)
