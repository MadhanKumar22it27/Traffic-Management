# import cv2
# import numpy as np

# def remove_raindrops(image_path, output_path):
#     """
#     Removes raindrops from an image using inpainting.

#     Args:
#         image_path (str): Path to the input image.
#         output_path (str): Path to save the output image.
#     """
#     try:
#         # Load the image
#         img = cv2.imread(image_path)

#         if img is None:
#             raise FileNotFoundError(f"Image not found at: {image_path}")

#         # Convert to grayscale for raindrop detection (if needed)
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         # Detect raindrops (This is the most challenging part and may require adjustments)
#         # Using a simple thresholding and contour detection as a starting point.
#         _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV) #adjust threshold values as needed
#         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         # Create a mask for inpainting
#         mask = np.zeros_like(gray)

#         # Draw contours on the mask
#         cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

#         # Refine the mask (optional, but often improves results)
#         kernel = np.ones((5, 5), np.uint8) #adjust kernel size as needed
#         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

#         # Inpaint the image
#         inpainted_img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA) #adjust inpaint radius as needed. cv2.INPAINT_NS may be another good choice

#         # Save the result
#         cv2.imwrite(output_path, inpainted_img)

#         print(f"Raindrops removed and saved to: {output_path}")

#     except Exception as e:
#         print(f"An error occurred: {e}")

# # Example usage
# input_image_path = "raindrop.jpg"  # Replace with your input image path
# output_image_path = "raindrop_removed.jpg" # Replace with your desired output path
# remove_raindrops(input_image_path, output_image_path)

import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os
from torchvision.utils import save_image
import torch.nn as nn

# ==== Define Generator Model (Same as Used in Training) ====
class GeneratorUNet(nn.Module):
    def __init__(self):
        super(GeneratorUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ==== Define Image Transformation ====
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize image to match training size
    transforms.ToTensor(),          # Convert image to tensor
])

# ==== Function to Remove Raindrops ====
def remove_raindrops(input_path, output_path, model_path="generator.pth"):
    # Check if the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found!")

    # Load Trained Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = GeneratorUNet().to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()

    # Load and Transform Image
    img_rainy = Image.open(input_path).convert("RGB")
    img_rainy_tensor = transform(img_rainy).unsqueeze(0).to(device)  # Add batch dimension

    # Generate Output
    with torch.no_grad():
        output = generator(img_rainy_tensor)

    # Save Output Image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_image(output, output_path)
    print(f"✅ Process completed! Rain-free image saved at: {output_path}")

# ==== Run the Script from Command Line ====
if __name__ == "__main__":
    remove_raindrops(r"input.png", r"Output\removed.png")
