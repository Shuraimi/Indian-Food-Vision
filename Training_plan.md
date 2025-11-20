# Complete Beginner's Guide: Training MobileNetV3 Food Classifier (80 Classes)
## Everything You Need to Know & Do (No Confusion)

**Target Audience**: Beginners and intermediate ML practitioners  
**Your Goal**: Train a production-ready food classifier that works on mobile phones  
**Your Dataset**: 80 food classes, ~150,000 images total  
**Your Model**: MobileNetV3 (fast, small, accurate)  
**Timeline**: 2-4 weeks to complete training  

---

## PART 1: WHAT YOU LEARNED FROM YOUR EXPERIMENTS

### Your Test Results Summary (Key Findings)

You ran 4 experiments with different models and augmentation:

| Model | With Augmentation? | Best Test Accuracy | Best Test Loss | Overfitting? |
|-------|-------------------|-------------------|----------------|------------|
| EfficientNet-Lite | No | 65.3% | 1.29 | Yes (after 15 epochs) |
| MobileNetV3 | No | 74.0% | 1.26 | Moderate |
| EfficientNet-Lite | Yes (TrivialAugment) | 68.8% | 1.16 | Yes (after 15 epochs) |
| MobileNetV3 | Yes (TrivialAugment) | **76.6%** | **1.01** | Slight (after 12 epochs) |

### What This Means (In Simple Terms)

1. **MobileNetV3 is your winner** â†’ Use this model going forward
2. **Data augmentation helps a lot** â†’ Your images need randomization (flips, rotations, color changes)
3. **Stop training at ~12 epochs** â†’ After 12 epochs, your model starts memorizing instead of learning
4. **Your model overfits after epoch 12** â†’ Train loss (near 0%) vs Test loss (increases) = memorization
5. **You need stronger regularization** â†’ Use weight decay, early stopping, and balanced loss

---

## PART 2: WHAT IS OVERFITTING? (UNDERSTAND THIS FIRST)

### Simple Explanation

Imagine you're studying for an exam:
- **Good learning**: You understand concepts, can answer any question
- **Overfitting**: You memorize the exact practice problems but can't solve new ones

**In neural networks**:
- **Good training**: Model learns general food features (color, shape, texture)
- **Overfitting**: Model memorizes your specific training images instead of learning patterns

### How to Detect Overfitting (Three Easy Signs)

**Sign 1: Look at Your Loss Curves**
```
GOOD (NO OVERFITTING):
Training Loss: 2.5 â†’ 1.0 â†’ 0.5 â†’ 0.4 (keeps going down)
Validation Loss: 2.6 â†’ 1.1 â†’ 0.6 â†’ 0.5 (keeps going down, similar to training)
Gap between them: Small (~0.1) âœ“

BAD (OVERFITTING):
Training Loss: 2.5 â†’ 1.0 â†’ 0.1 â†’ 0.01 (goes to nearly ZERO)
Validation Loss: 2.6 â†’ 1.1 â†’ 0.7 â†’ 0.9 (STARTS GOING UP!)
Gap: Large and growing (0.89) âœ—
```

**Sign 2: Check Your Accuracies**
```
GOOD:
Train Accuracy: 90% â†’ 92% â†’ 94% â†’ 95%
Test Accuracy:  88% â†’ 90% â†’ 92% â†’ 93%
Gap: ~2% (small) âœ“

BAD (YOUR CURRENT PROBLEM):
Train Accuracy: 90% â†’ 95% â†’ 97% â†’ 99%
Test Accuracy:  88% â†’ 90% â†’ 91% â†’ 91%
Gap: ~8% (growing) âœ—
```

**Sign 3: Check F1 Scores**
- If validation F1 score stops improving or goes down while training F1 keeps improving = overfitting

### Your Specific Problem (From Your Results)

Your MobileNetV3 experiment shows:
- Train accuracy near 98% âœ—
- Test accuracy stuck at ~76% âœ“
- This gap = **overfitting happening**
- Solution: Train for fewer epochs (~5-12 instead of 15)

---

## PART 3: HOW TO PREVENT OVERFITTING (Your Arsenal)

### Weapon 1: Weight Decay (Penalty for Large Weights)

**What it does**: Punishes the model if weights get too large. Large weights = overfitting risk.

**How to use it**:
```python
optimizer = optim.AdamW(model.parameters(), 
                        lr=0.001, 
                        weight_decay=0.0001)  # â† This is weight decay
```

**Rules**:
- If overfitting gap < 5%: `weight_decay=0.0001` (default)
- If overfitting gap 5-10%: `weight_decay=0.0001` (same)
- If overfitting gap > 10%: `weight_decay=0.0005` (increase it)
- If underfitting (both accuracies low): `weight_decay=0.00001` (decrease it)

### Weapon 2: Data Augmentation (Make Data Varied)

**What it does**: Randomly changes training images (rotate, flip, brighten) so model can't memorize exact images.

**Moderate augmentation (NOT TOO MUCH - for food images):**
```python
train_transform = A.Compose([
    # Random crop and resize
    A.RandomResizedCrop(224, 224, scale=(0.8, 1.0), p=1.0),
    
    # Flip horizontally (food looks similar flipped)
    A.HorizontalFlip(p=0.5),
    
    # Slight rotation (Â±20 degrees - food angle matters)
    A.Rotate(limit=20, p=0.3),
    
    # Color/brightness changes (lighting varies in real world)
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
    
    # Random dropout (hide parts of image)
    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
    
    # Normalize to ImageNet standard (MUST DO THIS)
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Validation: NO augmentation, only resize + normalize
val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
```

**Why this level?** 
- Heavy augmentation (extreme rotations, severe color shifts) can make food unrecognizable
- Light augmentation (only flips) not enough for food variety
- This is the "Goldilocks zone" âœ“

### Weapon 3: Early Stopping (Stop Before Overfitting)

**What it does**: Stops training when validation gets worse instead of continuing.

**How to implement**:
```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience  # How many bad epochs before stopping
        self.min_delta = min_delta  # Minimum improvement to count as "better"
        self.best_loss = float('inf')
        self.counter = 0
        self.best_model = None
    
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            # Validation improved
            self.best_loss = val_loss
            self.counter = 0
            self.best_model = model.state_dict().copy()
            return False  # Keep training
        else:
            # Validation not improving
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop training
        return False

# Usage in training loop:
early_stopping = EarlyStopping(patience=5)
for epoch in range(max_epochs):
    train_loss = train_one_epoch(...)
    val_loss = validate_epoch(...)
    
    if early_stopping(val_loss, model):
        print(f"Stopped at epoch {epoch} - no improvement for 5 epochs")
        break

# Restore best model
model.load_state_dict(early_stopping.best_model)
```

**Rules**:
- `patience=3-5` for small dataset (10% phase)
- `patience=5` for medium dataset (50% phase)
- `patience=5-7` for full dataset (100% phase)

### Weapon 4: Appropriate Learning Rate (Speed of Learning)

**What it does**: Controls how fast the model learns. Too fast = unstable, too slow = never learns.

**Learning rates by stage**:
```
Stage 1 (Frozen backbone, epoch 1-5): LR = 0.001
Stage 2 (Unfrozen backbone, epoch 6-12): LR = 0.0001 (10x lower!)
Stage 3 (Full unfrozen, epoch 13+): LR = 0.00005 (20x lower!)
```

**Why different rates?**
- Frozen backbone: Learning new task, use normal rate
- Unfrozen backbone: Don't want to break pre-trained weights, use very low rate
- Full unfrozen: Even more careful, use very very low rate

### Weapon 5: Fewer Epochs (Stop Training Earlier)

**From your experiments**: Best results at ~5 epochs, overfitting visible at 12+ epochs

**Rules**:
- 10% dataset: Max 5 epochs (training is very fast)
- 50% dataset: Max 10 epochs
- 100% dataset: Max 12 epochs with early stopping

---

## PART 4: STEP-BY-STEP TRAINING ROADMAP

### Phase 1: Train on 10% of Data (Validation Phase)
**Duration**: 1-2 days  
**Goal**: Make sure your code works and find the right settings

**Steps**:
1. Take only 10% of your images (~15,000 images)
2. Split into train/val (90% train, 10% val)
3. Set hyperparameters (see table below)
4. Train for max 5 epochs
5. Check learning curves
6. If curves look good, proceed to Phase 2

**Hyperparameters for Phase 1**:
```
Model: MobileNetV3-Large
Image Size: 224 Ã— 224
Batch Size: 64 (if GPU memory allows, else 32)
Learning Rate: 0.001
Weight Decay: 0.0001
Optimizer: AdamW
Epochs: 5 (with early stopping)
Augmentation: Yes (use code above)
Loss: CrossEntropyLoss (with class weights if imbalanced)
```

**What to check**:
- âœ“ Training runs without errors
- âœ“ Validation loss decreases every epoch
- âœ“ No NaN or Inf values
- âœ“ Test accuracy 30-50% (low is OK for 10%)

---

### Phase 2: Train on 50% of Data (Scaling Test)
**Duration**: 3-5 days  
**Goal**: Confirm settings work on larger dataset

**Steps**:
1. Take 50% of your images (~75,000 images)
2. Use same split (90% train, 10% val)
3. Use same hyperparameters as Phase 1
4. Train for max 10 epochs
5. Check if overfitting is controlled
6. If good, proceed to Phase 3

**What to check**:
- âœ“ Validation accuracy better than Phase 1 (should be 60-70%)
- âœ“ Overfitting gap controlled (val-train < 15%)
- âœ“ Learning curves smooth (not noisy)

**If problems occur**:
- Increasing gap = increase weight decay to 0.0005
- Noisy curves = increase batch size if possible
- Slow learning = increase learning rate slightly

---

### Phase 3: Train on 100% of Data (Final Model)
**Duration**: 5-7 days  
**Goal**: Train your final production model

**Steps**:
1. Use all 150,000 images
2. Same split (90% train, 10% val)
3. Same hyperparameters
4. Train for max 12 epochs with early stopping
5. Evaluate on separate test set
6. Save model for mobile deployment

**Expected Results**:
- Train Accuracy: 80-92%
- Test Accuracy: 75-82%
- Overfitting Gap: 5-15% (acceptable)

---

## PART 5: COMPLETE TRAINING SCRIPT (Copy-Paste Ready)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

# ============================================================================
# STEP 1: SET UP DATA AUGMENTATION (MODERATE - NOT TOO MUCH)
# ============================================================================

train_transform = A.Compose([
    A.RandomResizedCrop(224, 224, scale=(0.8, 1.0), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.3),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# ============================================================================
# STEP 2: CREATE DATASET CLASS
# ============================================================================

class FoodDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(image=img)['image']
        
        return img, label

# ============================================================================
# STEP 3: EARLY STOPPING CLASS
# ============================================================================

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_model = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model = model.state_dict().copy()
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# ============================================================================
# STEP 4: PROGRESSIVE UNFREEZING (FREEZE â†’ UNFREEZE LAST BLOCKS)
# ============================================================================

def freeze_backbone(model):
    """Freeze all backbone layers, keep only classifier trainable"""
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

def unfreeze_last_blocks(model):
    """Unfreeze last 2 convolutional blocks"""
    # Unfreeze last blocks
    for param in model.features[-2:].parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

# ============================================================================
# STEP 5: TRAINING FUNCTION
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

# ============================================================================
# STEP 6: VALIDATION FUNCTION
# ============================================================================

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

# ============================================================================
# STEP 7: MAIN TRAINING LOOP
# ============================================================================

def train_model(
    train_loader, 
    val_loader, 
    num_classes=80,
    num_epochs=5,
    batch_size=64,
    learning_rate=0.001,
    weight_decay=0.0001,
    device='cuda',
    unfreeze_at_epoch=5,
    patience=5
):
    """
    Main training function
    
    Args:
        train_loader: PyTorch DataLoader for training
        val_loader: PyTorch DataLoader for validation
        num_classes: Number of food classes (80 in your case)
        num_epochs: Maximum epochs to train
        batch_size: Batch size for training
        learning_rate: Learning rate
        weight_decay: L2 regularization coefficient
        device: 'cuda' or 'cpu'
        unfreeze_at_epoch: When to unfreeze backbone layers
        patience: Early stopping patience
    
    Returns:
        model: Trained model
        history: Training history (losses and accuracies)
    """
    
    # Initialize model
    model = models.mobilenet_v3_large(pretrained=True)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    model = model.to(device)
    
    # Freeze backbone initially
    freeze_backbone(model)
    
    # Loss function with class weights (if dataset is imbalanced)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler (cosine annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=5, 
        T_mult=2, 
        eta_min=1e-6
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience)
    
    # Track history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'lr': []
    }
    
    print(f"\n{'='*70}")
    print(f"Training on {device.upper()}")
    print(f"Model: MobileNetV3-Large | Classes: {num_classes}")
    print(f"Learning Rate: {learning_rate} | Weight Decay: {weight_decay}")
    print(f"Batch Size: {batch_size} | Max Epochs: {num_epochs}")
    print(f"Unfreeze backbone at epoch: {unfreeze_at_epoch}")
    print(f"{'='*70}\n")
    
    for epoch in range(num_epochs):
        # STAGE 1 vs STAGE 2 unfreezing
        if epoch == unfreeze_at_epoch:
            print(f"\n>>> UNFREEZING BACKBONE AT EPOCH {epoch} <<<")
            unfreeze_last_blocks(model)
            
            # Recreate optimizer with different LRs for different layers
            optimizer = optim.AdamW([
                {'params': model.features[-2:].parameters(), 'lr': learning_rate / 10},
                {'params': model.classifier.parameters(), 'lr': learning_rate}
            ], weight_decay=weight_decay)
            
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=5, T_mult=2, eta_min=1e-6
            )
            print("âœ“ Last 2 blocks unfrozen | LR reduced for backbone\n")
        
        # Train one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Print progress
        gap = val_loss - train_loss
        overfitting_status = "âœ“ OK" if gap < 0.2 else "âš  OVERFITTING"
        
        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Gap: {gap:.4f} {overfitting_status}")
        
        # Step scheduler
        scheduler.step()
        
        # Early stopping
        if early_stopping(val_loss, model):
            print(f"\n>>> EARLY STOPPING AT EPOCH {epoch+1} <<<")
            print(f"Validation loss not improving for {patience} epochs\n")
            model.load_state_dict(early_stopping.best_model)
            break
    
    return model, history

# ============================================================================
# STEP 8: PLOT TRAINING CURVES (TO DETECT OVERFITTING)
# ============================================================================

def plot_training_curves(history):
    """Plot training history to visualize overfitting"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training vs Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training vs Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    print("Curves saved to training_curves.png")

# ============================================================================
# STEP 9: EXAMPLE USAGE (HOW TO RUN THIS)
# ============================================================================

if __name__ == "__main__":
    # Example: Your data paths and labels
    # You need to provide:
    # - train_image_paths: List of paths to training images
    # - train_labels: List of class labels (0-79 for 80 classes)
    # - val_image_paths: List of paths to validation images
    # - val_labels: List of class labels
    
    # Create datasets
    train_dataset = FoodDataset(
        image_paths=train_image_paths,
        labels=train_labels,
        transform=train_transform
    )
    
    val_dataset = FoodDataset(
        image_paths=val_image_paths,
        labels=val_labels,
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True, 
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=64, 
        shuffle=False, 
        num_workers=4
    )
    
    # Train model (PHASE 1: 10% data, 5 epochs)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model, history = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=80,
        num_epochs=5,  # Phase 1
        batch_size=64,
        learning_rate=0.001,
        weight_decay=0.0001,
        device=device,
        unfreeze_at_epoch=5,  # Don't unfreeze in Phase 1
        patience=3
    )
    
    # Plot curves to check for overfitting
    plot_training_curves(history)
    
    # Save model
    torch.save(model.state_dict(), 'mobilenetv3_phase1.pth')
    print("Model saved!")
```

---

## PART 6: HOW TO USE THE SCRIPT STEP-BY-STEP

### Step 1: Prepare Your Data

```python
# Organize your images like this:
# data/
#   train/
#     apple/
#       img1.jpg
#       img2.jpg
#     banana/
#       img1.jpg
#   val/
#     apple/
#       val1.jpg
#     banana/
#       val1.jpg

from pathlib import Path

# Collect all image paths and labels
train_image_paths = []
train_labels = []
class_to_idx = {}
idx = 0

for class_folder in sorted(Path('data/train').iterdir()):
    if class_folder.is_dir():
        class_name = class_folder.name
        class_to_idx[class_name] = idx
        
        for img_path in class_folder.glob('*.jpg'):
            train_image_paths.append(str(img_path))
            train_labels.append(idx)
        
        idx += 1

# Same for validation
val_image_paths = []
val_labels = []

for class_folder in sorted(Path('data/val').iterdir()):
    if class_folder.is_dir():
        class_name = class_folder.name
        
        for img_path in class_folder.glob('*.jpg'):
            val_image_paths.append(str(img_path))
            val_labels.append(class_to_idx[class_name])

print(f"Training images: {len(train_image_paths)}")
print(f"Validation images: {len(val_image_paths)}")
print(f"Classes: {len(class_to_idx)}")
```

### Step 2: Install Required Libraries

```bash
pip install torch torchvision albumentations opencv-python numpy matplotlib scikit-learn
```

### Step 3: Run Training

```python
# Run the training script (copy-paste from PART 5)
# It will:
# 1. Create MobileNetV3 model
# 2. Freeze backbone initially
# 3. Train for 5 epochs (Phase 1)
# 4. Save training curves
# 5. Display results

# Output will look like:
# Epoch  1/5 | Train Loss: 2.3454 | Train Acc: 0.2134 | Val Loss: 2.2341 | Val Acc: 0.2456 | Gap: -0.0113 âœ“ OK
# Epoch  2/5 | Train Loss: 1.8234 | Train Acc: 0.4523 | Val Loss: 1.7654 | Val Acc: 0.4678 | Gap: -0.0580 âœ“ OK
# ...
```

### Step 4: Interpret the Results

**After Phase 1, check:**

1. **Loss decreasing?** âœ“ Good â†’ Continue to Phase 2
2. **Gap between train/val < 0.2?** âœ“ Good â†’ Continue to Phase 2
3. **Validation accuracy 30-50%?** âœ“ Expected for 10% data
4. **No NaN or errors?** âœ“ Code works properly

**If problems:**
- Loss not decreasing â†’ Learning rate too low, increase to 0.002
- Loss diverging (exploding) â†’ Learning rate too high, decrease to 0.0005
- Overfitting gap > 0.3 â†’ Increase weight_decay to 0.0005

### Step 5: Move to Phase 2 (50% Data)

```python
# Use 50% of data instead
# Keep same hyperparameters
# Max 10 epochs instead of 5

model, history = train_model(
    train_loader=train_loader_50pct,  # 50% data
    val_loader=val_loader_50pct,
    num_epochs=10,  # Increased
    unfreeze_at_epoch=5,  # Unfreeze at epoch 5
    patience=5
)
```

### Step 6: Move to Phase 3 (100% Data)

```python
# Use all data
# Same hyperparameters
# Max 12 epochs with early stopping

model, history = train_model(
    train_loader=train_loader_full,  # 100% data
    val_loader=val_loader_full,
    num_epochs=12,
    unfreeze_at_epoch=5,
    patience=5  # Will stop early if overfitting
)

# This is your FINAL MODEL
torch.save(model.state_dict(), 'mobilenetv3_final_model.pth')
```

---

## PART 7: CHECKLIST - WHAT TO DO NOW

### Week 1: Phase 1 Training
- [ ] Install libraries (torch, albumentations, etc.)
- [ ] Organize data into train/val folders
- [ ] Create image paths and labels lists
- [ ] Copy the training script from PART 5
- [ ] Run Phase 1 training on 10% data (5 epochs)
- [ ] Plot curves and check for overfitting
- [ ] Save the trained model

### Week 2: Phase 2 Training
- [ ] Prepare 50% of data
- [ ] Run Phase 2 training (10 epochs, unfreeze backbone)
- [ ] Monitor per-class accuracy if possible
- [ ] Check if overfitting improved
- [ ] Save the model

### Week 3-4: Phase 3 Training + Deployment
- [ ] Run Phase 3 training on 100% data (max 12 epochs)
- [ ] Evaluate on separate test set
- [ ] Export model to ONNX or TFLite for mobile
- [ ] Test on mobile device if possible
- [ ] Document final accuracy and model size

---

## PART 8: COMMON PROBLEMS & SOLUTIONS

### Problem 1: "RuntimeError: Expected 3D (unbatched) or 4D (batched) input to conv2d"
**Cause**: Image shape is wrong  
**Solution**: Check image paths are correct and images load properly
```python
# Test image loading
img = cv2.imread('data/train/apple/img1.jpg')
print(img.shape)  # Should be (H, W, 3)
```

### Problem 2: "Out of Memory (OOM)" Error
**Cause**: Batch size too large for GPU  
**Solution**: Reduce batch size
```python
# Change from batch_size=64 to batch_size=32
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

### Problem 3: Loss Not Decreasing at All
**Cause**: Learning rate too low or data problem  
**Solution**: Increase learning rate or check data
```python
# Try learning_rate=0.002 or 0.005
optimizer = optim.AdamW(..., lr=0.002)
```

### Problem 4: Loss Becomes NaN or Inf
**Cause**: Learning rate too high  
**Solution**: Decrease learning rate significantly
```python
# Try learning_rate=0.0005
optimizer = optim.AdamW(..., lr=0.0005)
```

### Problem 5: Very High Overfitting (Gap > 0.5)
**Cause**: Model memorizing, need more regularization  
**Solution**: 
- Increase weight_decay to 0.0005
- Use stronger augmentation
- Train fewer epochs

---

## PART 9: KEY TAKEAWAYS (Remember These)

1. **You found MobileNetV3 works best** â†’ Use it for everything
2. **Overfitting appears after 12 epochs** â†’ Stop training before that
3. **Use augmentation** â†’ It helps a lot (+5% accuracy in your tests)
4. **Weight decay prevents overfitting** â†’ Always use it
5. **Early stopping saves time** â†’ Don't wait for all epochs
6. **Progressive unfreezing is important** â†’ Freeze â†’ Unfreeze last blocks â†’ Optionally unfreeze all
7. **Monitor gaps between train/val** â†’ Gap > 20% = overfitting, fix it
8. **Plot curves to visualize problems** â†’ See the learning curve patterns
9. **Scale gradually: 10% â†’ 50% â†’ 100%** â†’ Don't jump to full data immediately
10. **Save best model** â†’ Keep checkpoints of good epochs

---

## PART 10: WHERE TO GET HELP (NEXT STEPS)

After you run training:

1. **If you see overfitting**: Look at the gap between train/val loss in curves
   - Post the curves here with message: "My gap is growing, here's my curve"
   - I'll tell you exactly what to adjust

2. **If you get errors**: Copy the full error message and post here
   - Include: error message + code line causing it
   - I'll help debug

3. **If you want to optimize further**: Post your Phase 1 results
   - I'll suggest specific hyperparameter changes

4. **If you need mobile deployment help**: Post your model metrics
   - I'll guide you through quantization and export

---

## REFERENCES TO LEARN MORE

- **MobileNetV3 Paper**: https://arxiv.org/abs/1905.02175
- **Albumentations Docs**: https://albumentations.ai/
- **PyTorch Transfer Learning**: https://pytorch.org/tutorials/
- **Food Image Classification**: Referenced research papers in conversation

---

## QUICK COMMAND REFERENCE

```python
# Copy-paste these commands

# 1. Install libraries
pip install torch torchvision albumentations opencv-python numpy matplotlib scikit-learn

# 2. Check GPU availability
import torch
print(torch.cuda.is_available())

# 3. Check CUDA memory
import torch
print(torch.cuda.memory_allocated() / 1e9)  # GB

# 4. Save model
torch.save(model.state_dict(), 'my_model.pth')

# 5. Load model
model = models.mobilenet_v3_large()
model.classifier[-1] = nn.Linear(1280, 80)
model.load_state_dict(torch.load('my_model.pth'))

# 6. Test on single image
img = cv2.imread('test.jpg')
img = transform(image=img)['image']
img = img.unsqueeze(0).to('cuda')
output = model(img)
pred_class = torch.argmax(output, dim=1)
```

---

## FINAL WORDS

This guide covers everything you need to know. **Don't get paralyzed by options** â€” just follow the steps in order:

1. **Phase 1** â†’ Train on 10%
2. **Phase 2** â†’ Train on 50%
3. **Phase 3** â†’ Train on 100%

That's it. Come back to this chat anytime with:
- Error messages
- Learning curves (screenshots)
- Questions about next steps
- Results from each phase

You've got this! ðŸš€
