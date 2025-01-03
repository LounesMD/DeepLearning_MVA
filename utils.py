import torch
import torchvision
import torch.nn as nn
import numpy as np
import os 
from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset


def get_images_from_path(folder_path):
    # Dictionary to hold images, with subfolders as keys and images sorted by index
    images_by_subfolder = defaultdict(list)
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.png') and 'img_' in file:
                try:
                    # Extract the numeric index x from the filename "img_x.png"
                    index = int(file.split('_')[1].split('.')[0])  # Extract x from "img_x.png"
                    # Get the relative subfolder path (excluding the base folder path)
                    subfolder_name = os.path.basename(root)
                    img_path = os.path.join(root, file)
                    # Use context manager to ensure the file is properly closed
                    with Image.open(img_path) as img:
                        # Append the image to the appropriate subfolder's list, sorted by index
                        images_by_subfolder[subfolder_name].append((index, img.convert('L')))
                except (ValueError, IndexError):
                    print(f"Skipping file with unexpected format: {file}")
    # Sort images within each subfolder by their index
    for subfolder, images in images_by_subfolder.items():
        # Sort by the numeric index (first element of the tuple)
        images_by_subfolder[subfolder] = [img for _, img in sorted(images, key=lambda x: x[0])]


    # Assuming images_by_subfolder is already populated
    # Create an empty list to store image data in the desired tensor shape
    tensor_images = []
    # Define the target image size (64x64)
    target_size = (64, 64)
    # Iterate through each subfolder
    for subfolder, images in images_by_subfolder.items():
        folder_images = []
        
        # Iterate through each image in the subfolder
        for img in images:
            # Resize the image to (64, 64)
            img_resized = img.resize(target_size)
            
            # Convert the image to a NumPy array and normalize to [0, 1] range
            img_array = np.array(img_resized) / 255.0  # Convert to float and normalize
            
            # Add a channel dimension (grayscale, so it's 1 channel)
            img_array = np.expand_dims(img_array, axis=-1)  # Shape becomes (64, 64, 1)
            
            # Add the image to the list
            folder_images.append(img_array)
        
        # Append the images from the current subfolder to the main list
        tensor_images.append(folder_images)
    # Convert the list to a NumPy array and then to a PyTorch tensor
    # Convert the list to a numpy array of shape (len(subfolders), num_images_per_subfolder, 64, 64, 1)
    tensor_images = np.array(tensor_images)
    # Convert to a PyTorch tensor (shape will be (batch_size, num_images, 1, 64, 64))
    tensor_images = torch.tensor(tensor_images)
    tensor_images = tensor_images.permute(0, 1, 4, 2, 3)
    return tensor_images


class ImageDataset(Dataset):
    def __init__(self, tensor_images):
        """
        Args:
            tensor_images (Tensor): A tensor of shape (num_subfolders, 15, 1, 64, 64)
        """
        self.tensor_images = tensor_images
        self.num_subfolders = tensor_images.shape[0]
        self.num_images_per_subfolder = tensor_images.shape[1]

    def __len__(self):
        # Return the number of subfolders
        return self.num_subfolders

    def __getitem__(self, idx):
        # Get the subfolder images
        subfolder_images = self.tensor_images[idx]

        # First element: None
        first_element = 0

        # Second element: First 10 images (index 0 to 9)
        second_element = subfolder_images[:10]

        # Third element: Last 5 images (index 10 to 14)
        third_element = subfolder_images[10:]

        # Return as a tuple
        return first_element, second_element, third_element
