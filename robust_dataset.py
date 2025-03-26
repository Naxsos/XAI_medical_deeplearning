import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
import os
import numpy as np

class RobustChestXrayDataset(Dataset):
    def __init__(self, image_dir, df, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        # Pre-filter to remove problematic files
        self.filtered_df = self._filter_valid_images(df)
        print(f"Original dataset size: {len(df)}")
        print(f"Filtered dataset size: {len(self.filtered_df)} (removed {len(df) - len(self.filtered_df)} invalid images)")
        
    def _filter_valid_images(self, df):
        valid_indices = []
        invalid_files = []
        
        for idx, row in df.iterrows():
            img_name = row['Image Index']
            img_path = os.path.join(self.image_dir, img_name)
            
            if not os.path.exists(img_path):
                invalid_files.append(f"{img_path} (missing)")
                continue
                
            try:
                # Attempt to open the image to verify it's valid
                with Image.open(img_path) as img:
                    # Just accessing a property forces PIL to parse the file
                    img.format
                valid_indices.append(idx)
            except Exception as e:
                invalid_files.append(f"{img_path} ({str(e)})")
        
        if invalid_files:
            print(f"Found {len(invalid_files)} invalid images:")
            for f in invalid_files[:10]:  # Show first 10
                print(f"  - {f}")
            if len(invalid_files) > 10:
                print(f"  - ... and {len(invalid_files) - 10} more")
                
        return df.iloc[valid_indices]
        
    def __len__(self):
        return len(self.filtered_df)
    
    def __getitem__(self, idx):
        img_name = self.filtered_df.iloc[idx]['Image Index']
        img_path = os.path.join(self.image_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # This should never happen because we pre-filtered
            print(f"Unexpected error loading {img_path}: {e}")
            # Create a blank image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
            
        # Get labels (binary classification: normal vs abnormal)
        label = 1 if self.filtered_df.iloc[idx]['Finding Label'] != 'No Finding' else 0
        
        # Return image ID along with image and label
        return image, torch.tensor(label, dtype=torch.float32), img_name

# Example usage:
"""
# Create datasets with robust handling
train_dataset = RobustChestXrayDataset(IMAGES_PATH_RESIZED, train_df, transform=transforms_dict['train'])
val_dataset = RobustChestXrayDataset(IMAGES_PATH_RESIZED, val_df, transform=transforms_dict['val'])
test_dataset = RobustChestXrayDataset(IMAGES_PATH_RESIZED, test_df, transform=transforms_dict['val'])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
""" 