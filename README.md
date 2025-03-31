# Self-Correcting Attention Mechanism for Medical Imaging

This repository contains a self-correcting attention mechanism that helps medical image classification models focus on relevant areas with human-like learning behavior.

## Overview

The self-correcting attention mechanism implements a human-like learning strategy:

1. If a classification is **correct**, the model is encouraged to maintain attention in the same region but to focus more narrowly on the most relevant area (increasing attention values at peaks, decreasing elsewhere).

2. If a classification is **wrong**, the model is encouraged to explore different regions of the image (looking elsewhere).

3. If the model starts being wrong after previously doing well, it can roll back to a previously successful attention pattern.

## Components

The implementation consists of three main components:

1. **AdaptiveAttentionMemory** (`adaptive_attention_memory.py`) - Stores and tracks attention maps for each image and their associated classification outcomes.

2. **SelfCorrectiveAttentionLoss** (`self_corrective_loss.py`) - Implements three core losses:
   - **Focus Loss**: When correct, narrows the attention to a smaller, more focused region
   - **Exploration Loss**: When wrong, encourages looking at different areas
   - **Stabilization Loss**: When consistently wrong, reverts to previously successful attention

3. **Train Function** (`train_with_adaptive_attention.py`) - Integrates the memory and loss modules into the training loop.

## Usage

Here's how to use the self-correcting attention mechanism:

```python
from adaptive_attention_memory import AdaptiveAttentionMemory
from self_corrective_loss import SelfCorrectiveAttentionLoss
from train_with_adaptive_attention import train_with_adaptive_attention

# Create the model (must support returning attention maps)
model = ResNetWithAttention(num_classes=10).to(device)

# Use the training function
trained_model, history, tracked_history = train_with_adaptive_attention(
    model=model,
    train_loader=train_loader,  # must yield (images, labels, image_ids)
    val_loader=val_loader,      # must yield (images, labels, image_ids)
    device=device,
    num_epochs=15,
    
    # Hyperparameters for the attention mechanism
    lambda_focus=0.3,       # Weight for focus loss
    lambda_explore=0.3,     # Weight for exploration loss
    lambda_stable=0.2,      # Weight for stabilization loss
    memory_size=3,          # Number of previous attention maps to store
    rollback_threshold=2,   # Consecutive errors before reverting
    
    # Optional: track a specific image's attention evolution
    track_image_id="some_image_id"
)
```

## Requirements for Model

Your model should:
1. Return both predictions and attention maps in its forward pass
2. Use attention maps with values in the range [0, 1]

Example model format:
```python
class ModelWithAttention(nn.Module):
    def forward(self, x):
        # ... model logic ...
        return logits, attention_maps
```

## Requirements for Data Loader

Your data loader should return tuples of:
1. Images
2. Labels 
3. Unique image identifiers

Example data loader:
```python
class DatasetWithIDs(Dataset):
    def __getitem__(self, idx):
        # ... dataset logic ...
        return image, label, image_id
```

## Hyperparameter Tuning

- **lambda_focus**: Control how much the model should focus on the current area after correct predictions
- **lambda_explore**: Control how much the model should explore new areas after incorrect predictions
- **lambda_stable**: Control how strongly the model should revert to previous attention patterns
- **memory_size**: Number of previous attention maps to store (higher = more history)
- **rollback_threshold**: How many consecutive errors before reverting (lower = more conservative) 