import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import cv2

def calculate_metrics(original_image_path, restored_image_path):
    # Load original and restored images
    original_image = cv2.imread(original_image_path)
    restored_image = cv2.imread(restored_image_path)

    # Check if images were loaded correctly
    if original_image is None:
        raise ValueError(f"Error: Could not read original image from {original_image_path}")
    if restored_image is None:
        raise ValueError(f"Error: Could not read restored image from {restored_image_path}")

    # Calculate PSNR
    psnr_value = psnr(original_image, restored_image)

    # Calculate SSIM with appropriate win_size
    smaller_side = min(original_image.shape[0], original_image.shape[1], restored_image.shape[0], restored_image.shape[1])
    win_size = min(7, smaller_side)  # Ensure win_size is at most the smaller side of the images
    multichannel = original_image.ndim > 2 and restored_image.ndim > 2  # Check if images are color (3 channels)

    ssim_value = ssim(original_image, restored_image, win_size=3, multichannel=multichannel)

    return psnr_value, ssim_value

if __name__ == "__main__":
    
    original_image_path = 'data/pixelated/bird.jpg'
    restored_image_path = 'data/corrected/bird_corrected.jpg'

    try:
        psnr_value, ssim_value = calculate_metrics(original_image_path, restored_image_path)
        print(f"PSNR: {psnr_value}")
        print(f"SSIM: {ssim_value}")
    except ValueError as e:
        print(e)

