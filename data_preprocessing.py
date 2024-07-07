import cv2
import os

# Ensure the output directory exists
os.makedirs('data/pixelated', exist_ok=True)

def create_pixelated_images(image_path, output_dir):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return
    img_name = os.path.basename(image_path).split('.')[0]

    # JPEG Compression
    cv2.imwrite(os.path.join(output_dir, f"{img_name}_jpeg10.jpg"), img, [int(cv2.IMWRITE_JPEG_QUALITY), 10])
    cv2.imwrite(os.path.join(output_dir, f"{img_name}_jpeg20.jpg"), img, [int(cv2.IMWRITE_JPEG_QUALITY), 20])

    # Downscale and Upscale
    for scale in [5, 6]:
        small = cv2.resize(img, (img.shape[1]//scale, img.shape[0]//scale), interpolation=cv2.INTER_NEAREST)
        upscaled = cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(output_dir, f"{img_name}_nn_{scale}x.jpg"), upscaled)

        small = cv2.resize(img, (img.shape[1]//scale, img.shape[0]//scale), interpolation=cv2.INTER_LINEAR)
        upscaled = cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(output_dir, f"{img_name}_bilinear_{scale}x.jpg"), upscaled)

# Define specific image files to process
image_files = ['data/images/4.jpg', 'data/images/6.jpg']

# Apply the function to the specified images
for img_file in image_files:
    create_pixelated_images(img_file, 'data/pixelated')
