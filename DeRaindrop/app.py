import cv2
import numpy as np

# Step 1: Read and preprocess the image
def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    # Convert to grayscale for detection purposes
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

# Step 2: Detect raindrops using color and edge detection
def detect_raindrops(img):
    # Convert image to HSV to detect bright areas (raindrops)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define a mask for bright areas (raindrops have higher brightness in certain lighting)
    lower_hsv = np.array([0, 0, 200], dtype=np.uint8)
    upper_hsv = np.array([180, 50, 255], dtype=np.uint8)
    bright_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    # Apply a Gaussian blur to smooth out noise and enhance the raindrops
    blurred = cv2.GaussianBlur(bright_mask, (5, 5), 0)
    
    # Perform edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Create an empty mask where we will draw detected raindrop contours
    mask = np.zeros_like(bright_mask)
    
    # Find contours based on edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and draw contours
    for contour in contours:
        area = cv2.contourArea(contour)
        if 100 < area < 1500:  # Adjust these values as needed to filter small and large objects
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    
    # Refine mask using morphological operations to improve the contours
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask

# Step 3: Raindrop removal using localized inpainting
def remove_raindrops(image, mask):
    # Refine mask by removing small isolated areas (likely noise)
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    
    # Inpaint using the mask
    inpainted_img = cv2.inpaint(image, cleaned_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return inpainted_img

# Step 4: Main function to execute the process
def remove_raindrops_from_image(image_path):
    # Preprocess the image
    image, gray = preprocess_image(image_path)
    
    # Detect raindrops
    mask = detect_raindrops(image)
    
    # Remove raindrops using inpainting
    result = remove_raindrops(image, mask)
    
    return result

# Step 5: Save the result
image_path = 'E:/technical/Projects/Traffic Management/DeRaindrop/raindrop.jpeg'  # Your image path
result = remove_raindrops_from_image(image_path)

# Save the result
cv2.imwrite('E:/technical/Projects/Traffic Management/DeRaindrop/raindrop_removed_advanced_final.jpeg', result)

# Show the results for verification
original = cv2.imread(image_path)
combined = np.hstack((original, result))
cv2.imshow('Original Image vs Raindrop Removed Image', combined)
cv2.waitKey(0)
cv2.destroyAllWindows()