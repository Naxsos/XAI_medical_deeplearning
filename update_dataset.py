import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class XrayDatasetWithIDs(Dataset):
    def __init__(self, df, base_dir, transform=None, label_column="Finding Label"):
        self.df = df
        self.base_dir = base_dir
        self.transform = transform
        self.label_column = label_column
        self.image_column = "Image Index"  # The column with image filenames
        
        # Create class mapping
        self.classes = sorted(df[label_column].unique())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        print(f"Class mapping: {self.class_to_idx}")
        
        # Report stats
        print(f"Dataset size: {len(df)} images")
        print(f"Class distribution: {df[label_column].value_counts().to_dict()}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Return image, label, and image_id to match the expected format"""
        row = self.df.iloc[idx]
        img_id = row[self.image_column]
        image_path = os.path.join(self.base_dir, img_id)
        
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Create a blank image as fallback (this helps to continue but will affect model performance)
            img = Image.new('RGB', (224, 224), color='black')
        
        label = self.class_to_idx[row[self.label_column]]
        
        if self.transform:
            img = self.transform(img)
            
        # Return three values as expected by the training function
        return img, label, img_id

# Example of how to use this dataset:
"""
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Load dataframe
df = pd.read_csv("your_data.csv")

# Create dataset
dataset = XrayDatasetWithIDs(
    df=df,
    base_dir="resized_images_2",  # Use the correct directory
    transform=transform
)

# Create dataloader
dataloader = DataLoader(
    dataset, 
    batch_size=32, 
    shuffle=True,
    num_workers=4
)

# Your dataloader now returns (images, labels, img_ids) as expected
""" 