import os
import cv2
import numpy as np

# Path to the folder containing your images
folder_path = 'input/incision/orig_dataall/train_images'
folder_path2 = 'input/incision/orig_dataall2/train_images'

# List all files in the folder
files = os.listdir(folder_path)

# Iterate through each file in the folder
for file_name in files:
    # Check if the file is an image (you might want to adjust this check based on your file naming conventions)
    if file_name.endswith(".png") or file_name.endswith(".jpg") or file_name.endswith(".jpeg"):
        # Read the image
        img = cv2.imread(os.path.join(folder_path, file_name))
        print(file_name)
        # Convert the image from float32 to uint8
        img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)

        # Save the converted image back to the folder
        cv2.imwrite(os.path.join(folder_path2, file_name), img_uint8)