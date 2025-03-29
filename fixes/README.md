# Fixing Sparsity Loss Convergence in Attention Model

This directory contains scripts to fix the sparsity loss convergence issue in your attention-based chest X-ray classification model.

## Problem

The original sparsity loss doesn't converge properly due to three main issues:

1. **Entropy-based formulation**: The current entropy-based sparsity loss (`attn_flat * torch.log(attn_flat)`) doesn't work well for attention maps that are already somewhat activated in specific regions.

2. **Artificial scaling**: The division by 1000 in the sparsity loss formula artificially reduces its magnitude, making the gradients too small to effectively influence the model.

3. **Low weight**: The `lambda_sparsity=0.001` is too small compared to the classification loss, making its contribution to the total loss negligible.

## Solution

The improved implementation fixes these issues by:

1. Using L1 regularization which directly encourages true sparsity
2. Removing the artificial division by 1000
3. Increasing the weight to a more meaningful value (0.1)

## How to Use

### Option 1: Simple Fix in Notebook

Add this to a cell in your notebook:

```python
# Improved sparsity loss implementation
class ImprovedSelfCorrectiveAttentionLoss(nn.Module):
    def __init__(self, lambda_consistency=0.1, lambda_sparsity=0.1):
        super(ImprovedSelfCorrectiveAttentionLoss, self).__init__()
        self.cls_loss = nn.BCELoss()  # Keep BCE for multi-label
        self.lambda_consistency = lambda_consistency
        self.lambda_sparsity = lambda_sparsity
    
    def forward(self, pred, target, attn_map, historical_attn=None):
        # Classification loss (primary objective)
        cls_loss = self.cls_loss(pred, target)
        
        # Initialize attention losses
        consistency_loss = torch.tensor(0.0).to(pred.device)
        sparsity_loss = torch.tensor(0.0).to(pred.device)
        
        # Improved sparsity loss using L1 norm to encourage true sparsity
        attn_flat = attn_map.view(attn_map.size(0), -1)
        # Normalize attention maps for each sample in batch
        attn_flat_normalized = attn_flat / (torch.sum(attn_flat, dim=1, keepdim=True) + 1e-8)
        # L1 regularization encourages sparsity more directly than negative entropy
        sparsity_loss = torch.mean(torch.sum(attn_flat_normalized, dim=1))
        
        # Consistency loss (if historical attention is available)
        if historical_attn is not None:
            valid_indices = [i for i, h in enumerate(historical_attn) if h is not None]
            
            if valid_indices:
                valid_hist = torch.stack([historical_attn[i] for i in valid_indices]).to(pred.device)
                valid_curr = attn_map[valid_indices]
                
                # For multi-label, consider prediction correct if all labels match
                correct_mask = (pred.round() == target).all(dim=1).float()[valid_indices].view(-1, 1, 1)
                incorrect_mask = 1 - correct_mask
                
                consistency_term = torch.abs(valid_curr - valid_hist) * correct_mask
                correction_term = torch.exp(-torch.abs(valid_curr - valid_hist)) * incorrect_mask
                
                consistency_loss = torch.mean(consistency_term + correction_term)
        
        # Total loss with weights
        total_loss = cls_loss + self.lambda_consistency * consistency_loss + self.lambda_sparsity * sparsity_loss
        
        return total_loss, cls_loss, consistency_loss, sparsity_loss
```

Then replace:

```python
criterion = SelfCorrectiveAttentionLoss(lambda_consistency=0, lambda_sparsity=0.001)
```

with:

```python
criterion = ImprovedSelfCorrectiveAttentionLoss(lambda_consistency=0.1, lambda_sparsity=0.1)
```

### Option 2: Using the Fix Files

1. Import the improved loss function:
   ```python
   from fixes.improved_loss import ImprovedSelfCorrectiveAttentionLoss
   ```

2. Replace your criterion with:
   ```python
   criterion = ImprovedSelfCorrectiveAttentionLoss(lambda_consistency=0.1, lambda_sparsity=0.1)
   ```

### Visualization

After training both models (original and improved), you can use the visualization tools to compare attention maps:

```python
from fixes.visualize_attention import compare_attention_maps

compare_attention_maps(
    model_original,  # Model trained with original loss
    model_improved,  # Model trained with improved loss
    val_loader,      # Data loader
    device,          # Device (cuda/cpu)
    num_samples=10,  # Number of samples to visualize
    save_dir='attention_comparison'
)
```

## Expected Results

With the improved sparsity loss, you should observe:

1. **Sparsity loss that properly converges** during training
2. **More focused attention maps** that highlight the most relevant areas of the X-ray images
3. **Potentially improved classification performance** due to more precise feature attention

The visualization tools will help you confirm these improvements in a qualitative way.

## Files in this Directory

- `improved_loss.py`: Implementation of the improved sparsity loss
- `apply_fix.py`: Example of how to apply the fix to your training loop
- `visualize_attention.py`: Tools to compare and visualize the effect of the improved loss
- `README.md`: This file with instructions and explanations 