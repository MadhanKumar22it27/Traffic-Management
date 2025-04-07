import cv2
import numpy as np
import os

# Step 1: Dark Channel Calculation
def dark_channel(image, size=15):
    dark_channel_img = cv2.min(cv2.min(image[:, :, 0], image[:, :, 1]), image[:, :, 2])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark_channel_img = cv2.erode(dark_channel_img, kernel)
    return dark_channel_img

# Step 2: Estimate Atmospheric Light
def atmospheric_light(image, dark_channel_img):
    [h, w] = image.shape[:2]
    num_pixels = h * w
    num_brightest = int(max(np.floor(num_pixels / 1000), 1))  # Top 0.1% brightest pixels
    
    dark_vec = dark_channel_img.ravel()
    image_vec = image.reshape(num_pixels, 3)
    
    indices = dark_vec.argsort()[::-1][:num_brightest]
    brightest_pixels = image_vec[indices]
    
    return np.mean(brightest_pixels, axis=0)  # Estimate the atmospheric light

# Step 3: Transmission Map Calculation
def transmission_estimate(image, atmospheric_light, omega=0.95, size=15):
    norm_image = np.empty_like(image, dtype=np.float32)
    for i in range(3):
        norm_image[:, :, i] = image[:, :, i] / atmospheric_light[i]
    
    transmission = 1 - omega * dark_channel(norm_image, size)
    return transmission

# Guided Filter for Transmission Refinement
def guided_filter(I, p, r, eps):
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    corr_I = cv2.boxFilter(I * I, cv2.CV_64F, (r, r))
    corr_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * I + mean_b
    return q

# Step 4: Recovering the Radiance (Dehazed Image)
def recover(image, transmission, atmospheric_light, t0=0.1):
    transmission = np.clip(transmission, t0, 1)  # Avoid division by zero
    
    radiance = np.empty_like(image, dtype=np.float32)
    for i in range(3):
        radiance[:, :, i] = (image[:, :, i] - atmospheric_light[i]) / transmission + atmospheric_light[i]
    
    # Scale the radiance back to [0, 255]
    return np.clip(radiance * 255.0, 0, 255).astype(np.uint8)

# Step 5: Dehazing the Image with Contrast Enhancement
def dehaze(image_path):
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} does not exist.")
        return None
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}.")
        return None

    image = image.astype(np.float32) / 255.0  # Normalize the image
    
    # Perform dehazing
    dark_channel_img = dark_channel(image)
    atmospheric_light_value = atmospheric_light(image, dark_channel_img)
    transmission = transmission_estimate(image, atmospheric_light_value)

    # Refine the transmission map using guided filter
    gray_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    transmission_refined = guided_filter(gray_image, transmission, r=40, eps=1e-3)

    dehazed_image = recover(image, transmission_refined, atmospheric_light_value)
    
    # Post-processing: Enhance contrast using CLAHE (optional)
    lab = cv2.cvtColor(dehazed_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Sharpen the image to improve clarity
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    sharpened = cv2.filter2D(final_image, -1, kernel_sharpening)
    
    # Debug information: Show intermediate steps
    # cv2.imshow('Dark Channel', (dark_channel_img * 255).astype(np.uint8))
    # cv2.imshow('Transmission Map', (transmission_refined * 255).astype(np.uint8))
    
    return sharpened

# Step 6: Displaying Results
if __name__ == "__main__":
    image_path = 'traffic_image.webp'  # Path to your traffic image
    dehazed_image = dehaze(image_path)
    
    if dehazed_image is not None:
        original_image = cv2.imread(image_path)
        cv2.imshow('Original Traffic Photo', original_image)
        cv2.imshow('Dehazed Traffic Photo', dehazed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        