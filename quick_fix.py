import os
from PIL import Image

# Create a blank replacement for the corrupted file
corrupt_file = 'resized_images_2/00004673_016.png'
print(f"Creating a blank replacement for {corrupt_file}")

# Create directory if it doesn't exist
os.makedirs(os.path.dirname(corrupt_file), exist_ok=True)

# Create a blank black image (224x224 is standard for many models)
blank_img = Image.new('RGB', (224, 224), color='black')
blank_img.save(corrupt_file)

print(f"Replaced corrupted file with blank image.")
print("This fix allows your code to continue running without errors.")
print("Note: This is a quick fix. For better model performance, consider:")
print("1. Using the RobustChestXrayDataset class to filter all corrupted images")
print("2. Checking for other corrupted images in your dataset") 