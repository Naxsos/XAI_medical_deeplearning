# binary_transfer_learning.py

# 1. Imports
import pandas as pd
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import os
import numpy as np

# 2. Load CSV
df = pd.read_csv("bbox_resized_filtered_images.csv")
print("Sample rows:", df.head())

# 3. Config
image_column = "Image Index"
label_column = "Finding Label"

# Convert to binary labels
df[label_column] = df[label_column].replace({
    'Atelectasis': 'Finding',
    'Cardiomegaly': 'Finding',
    'Effusion': 'Finding',
    'Infiltrate': 'Finding',
    'Mass': 'Finding',
    'Nodule': 'Finding',
    'Pneumonia': 'Finding',
    'Pneumothorax': 'Finding'
})

# Explicitly set as binary classification
num_classes = 2
print(f"Unique classes: {df[label_column].unique()}")
print(f"Class distribution:\n{df[label_column].value_counts()}")

batch_size = 32
epochs = 20
image_size = 224
base_dir = "resized_images"

# 4. Dataset
class SimpleImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.base_dir = base_dir
        self.transform = transform
        self.classes = sorted(df[label_column].unique())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        print(f"Class mapping: {self.class_to_idx}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.base_dir, row[image_column])
        img = Image.open(image_path).convert("RGB")
        label = self.class_to_idx[row[label_column]]
        if self.transform:
            img = self.transform(img)
        return img, label

# 5. Transforms
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# 6. Dataset and Split
dataset = SimpleImageDataset(df, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# 7. Model - Transfer Learning with ResNet18
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pre-trained model
model = models.resnet18(pretrained=True)

# Freeze the model parameters
for param in model.parameters():
    param.requires_grad = False

# Replace the final fully connected layer for binary classification
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)

model = model.to(device)

# 8. Loss and Optimizer
criterion = nn.CrossEntropyLoss()
# Only optimize the parameters of the new layers
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# 9. Training Loop
for epoch in range(epochs):
    # Training phase
    model.train()
    total_loss = 0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)

    train_accuracy = correct_train / total_train

    # Validation phase
    model.eval()
    correct_val = 0
    total_val = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)
            
            # Store predictions and labels for metrics
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_accuracy = correct_val / total_val

    # Calculate binary classification metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    true_pos = np.sum((all_preds == 1) & (all_labels == 1))
    false_pos = np.sum((all_preds == 1) & (all_labels == 0))
    true_neg = np.sum((all_preds == 0) & (all_labels == 0))
    false_neg = np.sum((all_preds == 0) & (all_labels == 1))
    
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")
    print(f"Train Acc: {train_accuracy:.2%} | Val Acc: {val_accuracy:.2%}")
    print(f"Precision: {precision:.2%} | Recall: {recall:.2%} | F1: {f1:.2%}")
    print("--------------------------------------------------------------")

# 10. Save the model
torch.save(model.state_dict(), 'binary_classifier_xray.pth')
print("Model saved to 'binary_classifier_xray.pth'")

# 11. Evaluate on a few examples
model.eval()
with torch.no_grad():
    for images, labels in list(val_loader)[:1]:  # Just use the first batch
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        
        for i in range(min(5, len(images))):  # Show first 5 examples
            print(f"Image prediction: {dataset.classes[preds[i]]}, "
                  f"Confidence: {probabilities[i][preds[i]]:.2%}, "
                  f"True label: {dataset.classes[labels[i]]}") 