import torch
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np

def train_with_self_correction_with_auc(model, train_loader, val_loader, device, num_epochs=15):
    """
    Train the model with self-corrective attention and calculate ROC AUC for each epoch
    """
    # Initialize attention memory
    memory = AttentionMemory(memory_size=3, alpha=0.7)
    
    # Loss and optimizer
    criterion = SelfCorrectiveAttentionLoss(lambda_consistency=0.15, lambda_sparsity=0.1)
    optimizer = optim.Adam([
        {'params': model.features.parameters(), 'lr': 0.00001},
        {'params': model.attention.parameters(), 'lr': 0.0001},
        {'params': model.classifier.parameters(), 'lr': 0.001}
    ])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # Track metrics
    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'train_acc': [], 'train_auc': [],  # Added train_auc
        'val_loss': [], 'val_acc': [], 'val_auc': [],  # Added val_auc
        'cls_loss': [], 'consistency_loss': [], 'sparsity_loss': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        epoch_cls_loss = 0.0
        epoch_consistency_loss = 0.0
        epoch_sparsity_loss = 0.0
        
        # For ROC AUC calculation
        all_train_preds = []
        all_train_labels = []
        
        for images, labels, img_ids in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images = images.to(device)
            labels = labels.to(device)
            
            # Get historical attention maps for batch
            historical_attn = memory.get_historical_attention(img_ids)
            
            # Forward pass
            optimizer.zero_grad()
            logits, attn_maps = model(images)
            logits = logits.squeeze()
            
            # Calculate loss
            loss, cls_loss, consistency_loss, sparsity_loss = criterion(
                logits, labels, attn_maps, historical_attn
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            epoch_cls_loss += cls_loss.item()
            epoch_consistency_loss += consistency_loss.item()
            epoch_sparsity_loss += sparsity_loss.item()
            
            # Calculate accuracy
            preds = (logits > 0.5).float()
            train_correct += (preds == labels).sum().item()
            
            # Store predictions and labels for ROC AUC calculation
            all_train_preds.extend(logits.detach().cpu().numpy())
            all_train_labels.extend(labels.detach().cpu().numpy())
            
            # Update attention memory
            memory.update(img_ids, attn_maps, logits, labels)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        # For ROC AUC calculation
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            for images, labels, img_ids in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                # Get historical attention
                historical_attn = memory.get_historical_attention(img_ids)
                
                # Forward pass
                logits, attn_maps = model(images)
                logits = logits.squeeze()
                
                # Calculate loss
                loss, _, _, _ = criterion(logits, labels, attn_maps, historical_attn)
                
                # Track metrics
                val_loss += loss.item()
                preds = (logits > 0.5).float()
                val_correct += (preds == labels).sum().item()
                
                # Store predictions and labels for ROC AUC calculation
                all_val_preds.extend(logits.detach().cpu().numpy())
                all_val_labels.extend(labels.detach().cpu().numpy())
                
                # Update memory even for validation (helps track consistency)
                memory.update(img_ids, attn_maps, logits, labels)
        
        # Calculate epoch metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_cls_loss = epoch_cls_loss / len(train_loader)
        avg_consistency_loss = epoch_consistency_loss / len(train_loader)
        avg_sparsity_loss = epoch_sparsity_loss / len(train_loader)
        train_accuracy = 100 * train_correct / len(train_loader.dataset)
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / len(val_loader.dataset)
        
        # Calculate ROC AUC scores
        try:
            train_auc = roc_auc_score(all_train_labels, all_train_preds)
            val_auc = roc_auc_score(all_val_labels, all_val_preds)
        except ValueError as e:
            # This can happen if all labels are of one class
            print(f"Error calculating ROC AUC: {e}")
            train_auc = 0.0
            val_auc = 0.0
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['train_auc'].append(train_auc)  # Add train AUC to history
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)
        history['val_auc'].append(val_auc)  # Add val AUC to history
        history['cls_loss'].append(avg_cls_loss)
        history['consistency_loss'].append(avg_consistency_loss)
        history['sparsity_loss'].append(avg_sparsity_loss)
        
        # Print metrics
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f} (Cls: {avg_cls_loss:.4f}, ' + 
              f'Consistency: {avg_consistency_loss:.4f}, Sparsity: {avg_sparsity_loss:.4f})')
        print(f'Train Accuracy: {train_accuracy:.2f}%, Train AUC: {train_auc:.4f}')  # Added Train AUC
        print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%, Val AUC: {val_auc:.4f}')  # Added Val AUC
        
        # Update scheduler
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model_self_corrective.pth')
            print('Model saved!')
        
        print('-' * 60)
    
    return model, history 