class AttentionMemory:
    def __init__(self, memory_size=5, alpha=0.7):
        """
        Args:
            memory_size: Number of previous epochs to consider
            alpha: Exponential moving average factor (higher = more weight to recent)
        """
        self.memory_size = memory_size
        self.alpha = alpha
        self.attention_history = {}  # Maps image_id -> list of attention maps
        self.correct_predictions = {}  # Maps image_id -> list of correctness flags
    
    def update(self, image_ids, attention_maps, predictions, labels):
        """Update memory with new attention maps"""
        for i, img_id in enumerate(image_ids):
            # Get current attention map and whether prediction was correct
            # Clone to avoid modifying the original tensor
            attn = attention_maps[i].detach().cpu().clone()
            
            # Make sure attention has 4 dimensions for consistent storage
            if len(attn.shape) == 2:  # [H, W]
                attn = attn.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            elif len(attn.shape) == 3:  # [C, H, W]
                attn = attn.unsqueeze(0)  # [1, C, H, W]
            
            # Check if prediction was correct - avoid modifying tensors
            pred_label = (predictions[i] > 0.5).float().item()
            true_label = labels[i].item()
            correct = (pred_label == true_label)
            
            # Initialize if first time seeing this image
            if img_id not in self.attention_history:
                self.attention_history[img_id] = []
                self.correct_predictions[img_id] = []
            
            # Add new data
            self.attention_history[img_id].append(attn)
            self.correct_predictions[img_id].append(correct)
            
            # Keep only most recent entries
            if len(self.attention_history[img_id]) > self.memory_size:
                new_history = self.attention_history[img_id][1:]  # Slice to create a new list
                self.attention_history[img_id] = new_history
                
                new_correct = self.correct_predictions[img_id][1:]  # Slice to create a new list
                self.correct_predictions[img_id] = new_correct
    
    def get_historical_attention(self, image_ids):
        """Get exponentially weighted historical attention maps"""
        batch_history = []
        
        for img_id in image_ids:
            if img_id not in self.attention_history or not self.attention_history[img_id]:
                # No history available
                batch_history.append(None)
                continue
            
            # Get history for this image
            history = self.attention_history[img_id]
            correctness = self.correct_predictions[img_id]
            
            if len(history) == 1:
                # Only one entry in history
                batch_history.append(history[0])
                continue
            
            # Calculate weighted average, giving more weight to:
            # 1. More recent attention maps
            # 2. Attention maps from correct predictions
            weights = []
            for i, is_correct in enumerate(correctness):
                # Position weight (more recent = higher weight)
                pos_weight = self.alpha ** (len(correctness) - i - 1)
                # Correctness weight (correct predictions get higher weight)
                correct_weight = 1.2 if is_correct else 0.8
                weights.append(pos_weight * correct_weight)
            
            # Normalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            # Calculate weighted attention - without in-place operations
            weighted_attn = torch.zeros_like(history[0])
            for i, attn in enumerate(history):
                weighted_attn = weighted_attn + weights[i] * attn
            
            batch_history.append(weighted_attn)
        
        return batch_history 