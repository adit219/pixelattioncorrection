import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

def correct_pixelation(image_path, output_path):
    # Check if the image file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Read the image in color
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # Check if the image was read correctly
    if img is None:
        raise ValueError(f"Failed to read the image: {image_path}")

    print(f"Original image shape: {img.shape}")
    
    # Apply non-local means denoising for color images
    denoised_img = cv2.fastNlMeansDenoisingColored(img, None, 30, 30, 7, 21)

    print(f"Denoised image shape: {denoised_img.shape}")
    
    # Estimate noise variance for adaptive bilateral filter
    noise_var = estimate_noise_variance(denoised_img)
    
    # Adjust bilateral filter parameters based on estimated noise variance
    sigma_color = 75
    sigma_space = 75
    
    if noise_var is not None:
        sigma_color = 10 * noise_var
        sigma_space = 10 * noise_var

    # Apply a bilateral filter to enhance the image further
    enhanced_img = cv2.bilateralFilter(denoised_img, 9, sigma_color, sigma_space)

    print(f"Enhanced image shape: {enhanced_img.shape}")
    
    # Save the corrected image
    cv2.imwrite(output_path, enhanced_img)
    print(f"Corrected image saved at: {output_path}")

    # Visualize images
    visualize_images(img, denoised_img, enhanced_img)

def visualize_images(original, denoised, enhanced):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Denoised Image")
    plt.imshow(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Enhanced Image")
    plt.imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()

def estimate_noise_variance(image):
    # Estimate noise variance assuming Gaussian noise
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    noise = np.abs(gray_image.astype(np.float32) - cv2.fastNlMeansDenoising(gray_image, None, 10, 7, 21))
    noise_var = np.var(noise)
    return noise_var


input_image_path = 'data/pixelated/11.jpg'
output_image_path = 'data/corrected/8_corrected.jpg'
os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
correct_pixelation(input_image_path, output_image_path)
