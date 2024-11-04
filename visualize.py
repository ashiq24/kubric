from PIL import Image
import numpy as np

# Load the TIFF file
tiff_file_path = './output/depth_00000.tiff'  # Replace with your TIFF file path
png_file_path = 'output_image.png'  # Specify the output PNG file path

# Open the TIFF file
image = Image.open(tiff_file_path)

# Convert the image to a NumPy array
image_array = np.array(image)

# Normalize the array to the range 0-255
min_val = np.min(image_array)
max_val = np.max(image_array)
scaled_image_array = 255 * (image_array - min_val) / (max_val - min_val)
scaled_image_array = scaled_image_array.astype(np.uint8)

# Convert the scaled array back to an image
scaled_image = Image.fromarray(scaled_image_array, mode='L')  # Use 'L' for grayscale

# Save the image as PNG
scaled_image.save(png_file_path, format='PNG')

print(f"Image saved as {png_file_path}")