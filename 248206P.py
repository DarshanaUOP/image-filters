import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# mean filter
def mean_filter(image, kernel_size=3):
    padded_image = np.pad(image, pad_width=((kernel_size//2,), (kernel_size//2,), (0,)), mode='wrap')
    output_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            output_image[i, j] = np.mean(region, axis=(0, 1))
    return output_image

# median filter
def median_filter(image, kernel_size=3):
    padded_image = np.pad(image, pad_width=((kernel_size//2,), (kernel_size//2,), (0,)), mode='wrap')
    output_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            output_image[i, j] = np.median(region, axis=(0, 1))
    return output_image

# midpoint filter
def midpoint_filter(image, kernel_size=3):
    padded_image = np.pad(image, pad_width=((kernel_size//2,), (kernel_size//2,), (0,)), mode='wrap')
    output_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            max_val = np.max(region, axis=(0, 1))
            min_val = np.min(region, axis=(0, 1))
            output_image[i, j] = (max_val + min_val) / 2
    return output_image

# read files
home_directory = os.path.expanduser("sample-images")
for filename in os.listdir(home_directory):

    if filename.lower().endswith(".jpeg"): # get JPEG files only for processing, save results as JPG for avoid read results back.
        print(f"processing: {filename}")

        # reading the original image
        image_path = os.path.join(home_directory, filename)
        image = cv2.imread(image_path)
        
        # Apply filters
        mean_filtered = mean_filter(image)
        median_filtered = median_filter(image)
        midpoint_filtered = midpoint_filter(image)
        
        # Save filtered images
        cv2.imwrite(os.path.join(home_directory, f"{filename.split('.')[0]}_mean.jpg"), mean_filtered)
        cv2.imwrite(os.path.join(home_directory, f"{filename.split('.')[0]}_median.jpg"), median_filtered)
        cv2.imwrite(os.path.join(home_directory, f"{filename.split('.')[0]}_midpoint.jpg"), midpoint_filtered)

        # print images 
        plt.subplot(4, 2, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(4, 2, 2)
        plt.imshow(mean_filtered)
        plt.title('Mean filtered Image')
        plt.axis('off')

        plt.subplot(4, 2, 3)
        plt.imshow(median_filtered)
        plt.title('Median filtered Image')
        plt.axis('off')

        plt.subplot(4, 2, 4)
        plt.imshow(midpoint_filtered)
        plt.title('Midpoint filtered Image')
        plt.axis('off')

        plt.tight_layout()
        plt.show()
print("Filtering complete.")
