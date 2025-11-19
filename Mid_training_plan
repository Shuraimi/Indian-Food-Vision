# Deep Learning Image Classification Training Guide
## Food Classification (80 Classes, 150k Images) for Mobile Deployment

**Target Audience**: Advanced ML Engineer/Graduate Student  
**Framework**: PyTorch | **Models**: MobileNetV3, EfficientNet-Lite  
**Goal**: Production-ready model with minimal overfitting, optimal for mobile deployment

---

## Table of Contents
1. Model Selection Rationale
2. Comprehensive Training Roadmap (10% â†’ 50% â†’ 100%)
3. Hyperparameter Specifications
4. Class Imbalance & Loss Functions
5. Learning Curve Interpretation & Overfitting Prevention
6. Layer Unfreezing Strategy
7. Data Augmentation Pipeline
8. Evaluation Metrics & Model Quality Assessment
9. Mobile Deployment Optimization
10. Clean Implementation Checklist
11. Learning Resources & Implementation Order

---

## 1. Model Selection Rationale

### Why MobileNetV3 vs EfficientNet-Lite?

| Aspect | MobileNetV3 | EfficientNet-Lite |
|--------|-----------|------------------|
| **Parameters** | 5.4M (Large) / 2.5M (Small) | 3.5M â€“ 15M (B0â€“B5) |
| **ImageNet Acc** | 75.2% (V3-L), 67.4% (V3-S) | 80.4% â€“ 88% (B0â€“B7) |
| **Mobile Latency** | 0.8â€“1.2 ms (high-end phones) | 1.5â€“3 ms (comparable) |
| **Model Size (TFLite)** | ~9â€“14 MB | ~12â€“30 MB |
| **Training Speed** | Fast (3â€“4x faster than ResNet) | Fast, efficient scaling |
| **Fine-tuning** | Excellent | Excellent |
| **Recommendation for Your Task** | **Start here (V3-Large)** | Good alternative if accuracy is critical |

**Decision Tree**:
- **Priority: Speed + Simplicity** â†’ MobileNetV3-Large
- **Priority: Accuracy + Still Mobile-Friendly** â†’ EfficientNet-Lite B2/B3
- **Priority: Lightweight Edge Devices** â†’ MobileNetV3-Small

**For 80 classes with 150k images**: Start with **MobileNetV3-Large** (well-balanced), scale to **EfficientNet-Lite B2** if validation accuracy plateaus.

---

## 2. Comprehensive Training Roadmap: 10% â†’ 50% â†’ 100%

### Why This Approach?
- **Early detection of issues**: Catch overfitting, bugs, hyperparameter problems on 10% before wasting time on full dataset
- **Rapid iteration**: 10% trains in ~2â€“4 hours (vs 20â€“30 for 100%)
- **Validate scaling assumptions**: Confirm that hyperparameters work across dataset sizes
- **Progressive refinement**: Gradually increase data volume with proven settings

### Phase 1: 10% Dataset (Quick Validation) â€“ Target: 4â€“6 hours total

**Dataset**: ~15k images (~18â€“20 images per class)

**Objectives**:
- Verify code pipeline works (no bugs, data loading correct)
- Establish baseline hyperparameters
- Detect severe overfitting
- Identify if data augmentation helps
- Estimate training time for 100%

**Training Config**:
```python
image_size: 224  # Standard for MobileNetV3, EfficientNet
batch_size: 64
learning_rate: 0.001  # Start conservative
epochs: 30  # Quick pass, use early stopping
weight_decay: 0.0001
optimizer: AdamW
scheduler: CosineAnnealingWarmRestarts(T_0=10, T_mult=2)
warmup_epochs: 2
```

**Expected Outcome**:
- Training accuracy: 50â€“65%
- Validation accuracy: 35â€“50%
- Gap = overfitting signal (expected at small scale, monitor trend)

**Success Criteria**:
- No NaN/Inf loss
- Validation loss decreasing (not diverging immediately)
- Early stopping triggers around epoch 25â€“28

---

### Phase 2: 50% Dataset (Scaling Test) â€“ Target: 8â€“12 hours

**Dataset**: ~75k images (~940 images per class)

**Objectives**:
- Test if hyperparameters scale
- Diagnose class imbalance handling
- Refine learning rate & weight decay
- Finalize augmentation pipeline
- Measure accuracy improvement with more data

**Training Config** (Same as Phase 1, or adjust based on Phase 1 results):
```python
image_size: 224
batch_size: 64
learning_rate: 0.0008  # Slight decrease if Phase 1 had high loss volatility
epochs: 35
weight_decay: 0.0001
optimizer: AdamW
scheduler: CosineAnnealingWarmRestarts(T_0=12, T_mult=2)
warmup_epochs: 2
```

**Expected Outcome**:
- Training accuracy: 70â€“80%
- Validation accuracy: 60â€“70%
- Overfitting gap: 10â€“15% (improvement vs Phase 1)

**Validation**:
- Plot train/val loss curves: Should show smooth convergence, validation loss decreasing
- Check per-class F1 scores: Minority classes should improve
- Early stopping: Around epoch 30â€“33

---

### Phase 3: 100% Dataset (Final Production Model) â€“ Target: 18â€“28 hours

**Dataset**: ~150k images (~1,875 images per class average)

**Objectives**:
- Train production model
- Final hyperparameter tuning
- Achieve target accuracy
- Prepare for quantization & mobile deployment

**Training Config** (Based on Phase 2 validation):
```python
image_size: 224
batch_size: 64  # If GPU memory allows, consider 128 for faster convergence
learning_rate: 0.0008
epochs: 40  # Increase slightly; more data = longer convergence
weight_decay: 0.0001
optimizer: AdamW
scheduler: CosineAnnealingWarmRestarts(T_0=15, T_mult=2)
warmup_epochs: 3  # Longer warmup for large dataset
early_stopping_patience: 5  # Epochs without improvement before stopping
```

**Expected Outcome**:
- Training accuracy: 85â€“92%
- Validation accuracy: 75â€“85%
- Overfitting gap: 8â€“12% (manageable)

---

## 3. Ideal Hyperparameters (Based on Research & Best Practices)

### Image Size
| Model | Standard | Mobile-Optimized | Trade-off |
|-------|----------|-----------------|-----------|
| **MobileNetV3** | 224Ã—224 | 192Ã—192 | 224 preferred (0.3â€“0.5% acc gain) |
| **EfficientNet-Lite** | 224Ã—224 | 192Ã—192 | Depends on variant (B0â€“B5) |

**Recommendation**: Use **224Ã—224** for training (ImageNet standard). For mobile, can test 192Ã—192 post-training.

### Batch Size
- **GPU Memory Consideration**: MobileNetV3 @ 224Ã—224 â‰ˆ 4â€“6 GB for batch_size=64
- **Optimal Range**: 32â€“64 (powers of 2)
  - **64**: Faster training, stable gradient updates
  - **32**: More stable if OOM occurs
  - **128**: Only if 16+ GB VRAM available

**Recommendation**: Start with **batch_size=64**; reduce to 32 if OOM.

### Learning Rate Schedule

**Recommended Strategy**: **LinearWarmup + CosineAnnealing** (most stable for mobile models)

```python
# Phase 1 & 2:
scheduler = CosineAnnealingWarmRestarts(T_0=10, T_mult=2)
warmup_epochs = 2

# Phase 3:
scheduler = CosineAnnealingWarmRestarts(T_0=15, T_mult=2)
warmup_epochs = 3
```

**Rationale**:
- Warmup (0 â†’ 0.001 over 2â€“3 epochs): Stabilizes early training
- Cosine Annealing: Smooth LR decay, empirically best for transfer learning
- T_mult=2: Restarts get longer, prevents getting stuck in local minima

**Learning Rate Values**:
| Stage | Frozen Backbone | Fine-tuning (Unfrozen) |
|-------|-----------------|----------------------|
| **Initial LR** | 0.001 | 0.0001â€“0.0003 |
| **Min LR** | 1e-6 | 1e-7 |

### Weight Decay (L2 Regularization)
- **Standard**: 0.0001 (1e-4) â€“ prevents overfitting
- **If heavy overfitting persists**: Increase to 0.0005â€“0.001
- **If underfitting**: Decrease to 1e-5 or remove

**Recommendation**: Start **0.0001**, monitor Phase 1 results.

### Number of Epochs
| Dataset Size | Min Epochs | Max Epochs | Early Stopping Patience |
|---|---|---|---|
| 10% | 25 | 40 | 3â€“5 |
| 50% | 30 | 50 | 5â€“7 |
| 100% | 35 | 60 | 5â€“7 |

**Note**: Use **early stopping** (patience=5â€“7 epochs without improvement) to avoid wasteful training.

---

## 4. Handling Class Imbalance & Loss Functions

### Diagnosis: Is Your Dataset Imbalanced?

```python
# Check class distribution
from collections import Counter
class_counts = Counter(train_labels)
print(sorted(class_counts.items()))
ratio = max(class_counts.values()) / min(class_counts.values())
print(f"Imbalance ratio: {ratio:.2f}x")
```

- **Ratio < 2**: Slight imbalance, standard CrossEntropyLoss sufficient
- **Ratio 2â€“5**: Moderate, use weighted loss or focal loss
- **Ratio > 5**: Severe, combine weighted loss + augmentation

### Solution 1: Weighted CrossEntropyLoss (Recommended for moderate imbalance)

```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch

# Compute balanced class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Use in loss function
criterion = nn.CrossEntropyLoss(weight=weights)
```

**Why It Works**: Down-weights majority classes, up-weights minority classes.

### Solution 2: Focal Loss (For severe imbalance)

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.CE = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, logits, targets):
        ce_loss = self.CE(logits, targets)
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            focal_loss *= self.alpha[targets]
        
        return focal_loss.mean()

# Usage
alpha = torch.tensor(class_weights).to(device)
criterion = FocalLoss(alpha=alpha, gamma=2)
```

**Why It Works**: (1 - p_t)^gamma down-weights easy samples, focuses on hard negatives.

### Solution 3: Combination (Weighted Sampling)

```python
# Oversample minority classes in DataLoader
from torch.utils.data import WeightedRandomSampler

sample_weights = 1.0 / class_counts[train_labels]
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_labels),
    replacement=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    sampler=sampler  # Ensures balanced batches
)
```

### Recommendation for Your Task (80 Classes, 150k Images)

**Priority Order**:
1. **Start with Weighted CrossEntropyLoss** (simplest, often sufficient)
2. **Monitor per-class F1 scores** during Phase 1
3. **If minority classes perform poorly** (F1 < 0.60), switch to Focal Loss
4. **Combine with augmentation** (see Section 7) for best results

---

## 5. Reading Learning Curves & Detecting Overfitting

### Plot Template (PyTorch)

```python
import matplotlib.pyplot as plt

def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    
    # Loss curves
    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(val_losses, label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid()
    ax1.set_title('Training vs Validation Loss')
    
    # Accuracy curves
    ax2.plot(train_accs, label='Train Acc', linewidth=2)
    ax2.plot(val_accs, label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid()
    ax2.set_title('Training vs Validation Accuracy')
    
    plt.tight_layout()
    plt.show()

# Track during training
plot_training_curves(history['train_loss'], history['val_loss'], 
                     history['train_acc'], history['val_acc'])
```

### Interpretation Patterns

| Pattern | Diagnosis | Action |
|---------|-----------|--------|
| Both curves decreasing, small gap (<5%) | **Ideal (No overfitting)** | Continue training |
| Train loss â†’ 0, val loss plateaus/increases | **Overfitting** | Early stop, add regularization |
| Both curves flat/increasing | **Underfitting** | Increase LR, reduce regularization |
| Erratic/noisy curves | **Batch size too small OR LR too high** | Increase batch size, reduce LR |
| Val loss diverges after few epochs | **LR too high OR bad initialization** | Reduce LR, check loss scale |

### Quantitative Thresholds

```python
def diagnose_training(train_loss, val_loss, epoch_threshold=10):
    """
    Diagnose overfitting based on loss trajectory
    """
    gap = val_loss[-1] - train_loss[-1]
    
    if gap < 0.05:
        return "No overfitting - Continue"
    elif gap < 0.15:
        return "Slight overfitting - Monitor & consider early stopping"
    elif gap > 0.3:
        return "SEVERE overfitting - Stop immediately, add regularization"
    else:
        return "Moderate overfitting - Apply regularization or reduce epochs"

# Usage
print(diagnose_training(train_losses, val_losses))
```

### Early Stopping Implementation

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_model_state = None
    
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict().copy()
        else:
            self.counter += 1
        
        return self.counter >= self.patience
    
    def restore_best_model(self, model):
        model.load_state_dict(self.best_model_state)
        return model

# In training loop
early_stopping = EarlyStopping(patience=5, min_delta=0.001)
for epoch in range(max_epochs):
    train_loss = train_epoch(...)
    val_loss = validate_epoch(...)
    
    if early_stopping(val_loss, model):
        print(f"Early stopped at epoch {epoch}")
        model = early_stopping.restore_best_model(model)
        break
```

---

## 6. When & How to Unfreeze Layers

### Strategy: Progressive Unfreezing

**Phase 1 (Epochs 1â€“5): Frozen Backbone**
- Keep all base model layers frozen
- Train only the classification head (final Dense layer)
- Rationale: Learn task-specific patterns without corrupting ImageNet features

```python
# Freeze backbone
for param in model.base_model.parameters():
    param.requires_grad = False

# Only unfreeze head
for param in model.classifier.parameters():
    param.requires_grad = True

# Lower learning rate for head
optimizer = optim.AdamW([
    {'params': model.classifier.parameters(), 'lr': 0.001}
], weight_decay=0.0001)
```

**Phase 2 (Epochs 6â€“15): Unfreeze Last Block**
- If validation accuracy plateaus after 5 epochs with frozen backbone, unfreeze the last convolutional block
- Rationale: Fine-tune top-level features to your specific food classes

```python
# For MobileNetV3: unfreeze last 2 blocks
for param in model.base_model[-2:].parameters():  # Last 2 blocks
    param.requires_grad = True

# Use different LRs for different layers
optimizer = optim.AdamW([
    {'params': model.base_model[-2:].parameters(), 'lr': 0.0001},
    {'params': model.classifier.parameters(), 'lr': 0.001}
], weight_decay=0.0001)

# Recompile scheduler with new optimizer
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
```

**Phase 3 (Epochs 16+): Full Fine-tuning (Optional)**
- Only if Phase 2 still shows improvement and you have >30 epochs
- Unfreeze entire model with very low learning rate

```python
# Unfreeze all layers
for param in model.parameters():
    param.requires_grad = True

# Use very low LR for pre-trained layers
optimizer = optim.AdamW([
    {'params': model.base_model.parameters(), 'lr': 0.00005},  # Very low
    {'params': model.classifier.parameters(), 'lr': 0.0005}    # Slightly higher
], weight_decay=0.0001)
```

### Decision Flowchart

```
â”Œâ”€ Start with frozen backbone
â”‚
â”œâ”€ Epoch 5: Check val_acc
â”‚
â”œâ”€ If val_acc improving & < target â†’ Stay frozen
â”‚
â”œâ”€ If val_acc plateaued â†’ Unfreeze last block
â”‚         â†“
â”‚     â”œâ”€ Epoch 15: Check again
â”‚     â”‚
â”‚     â”œâ”€ If improving â†’ Keep unfrozen, continue
â”‚     â”‚
â”‚     â””â”€ If plateaued â†’ Consider full unfreezing
â”‚
â””â”€ Final: Always use lower LR for pre-trained layers
```

### Best Practice: Two-Stage Training

```python
# Stage 1: Frozen backbone (5 epochs)
train_stage_1(model, train_loader, val_loader, epochs=5)

# Stage 2: Unfreeze + fine-tune (20 epochs)
for param in model.base_model[-2:].parameters():
    param.requires_grad = True

train_stage_2(model, train_loader, val_loader, epochs=20)
```

---

## 7. Data Augmentation Pipeline (Best Practices)

### Why Augmentation Matters
- **Reduces overfitting** by preventing memorization
- **Improves generalization** to real-world variations
- **Balances minority classes** (visual augmentation)

### Recommended Pipeline (Using Albumentations)

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Training augmentation (aggressive)
train_transform = A.Compose([
    # 1. Resize/Crop
    A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), p=1.0),
    
    # 2. Geometric transforms
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.3),
    A.Affine(scale=(0.9, 1.1), translate_percent=0.1, p=0.3),
    
    # 3. Dropout-based regularization
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
    
    # 4. Color/brightness shifts (important for food images)
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    
    # 5. Normalization (last step)
    A.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], bbox_params=None)

# Validation augmentation (minimal, only normalization + resize)
val_transform = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Usage in dataset
class FoodDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        
        return img, self.labels[idx]
```

### Augmentation Strategy by Dataset Size

| Size | Aggressive | Reason |
|------|-----------|--------|
| **10%** | Very High (all augmentations) | Few samples, need diversity |
| **50%** | Medium-High | Still room for augmentation |
| **100%** | Medium (reduce dropout/color jitter) | More natural diversity, avoid over-augmentation |

### Key Food-Specific Augmentations

```python
# For food images, these are especially helpful:
A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),  # Lighting variations
A.GaussNoise(p=0.2),                                                    # Camera noise
A.MotionBlur(blur_limit=7, p=0.2),                                      # Motion blur
A.Perspective(scale=(0.05, 0.1), p=0.3),                                # Viewing angle
```

### DO NOT Over-Augment

**Avoid in training**:
- Extreme rotations (>45Â°) â€“ food orientation is meaningful
- Aggressive color shifts (>0.5 saturation) â€“ food color is important
- Heavy compression/noise â€“ may hide food details

**Rule of Thumb**: If augmented images don't look like real food anymore, reduce augmentation intensity.

---

## 8. Evaluation Metrics & Model Quality Assessment

### Recommended Metrics for 80-Class Food Classification

```python
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

def evaluate_model(y_true, y_pred, y_pred_proba=None):
    """Comprehensive model evaluation"""
    
    # 1. Accuracy (overall)
    acc = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {acc:.4f}")
    
    # 2. F1 Scores (macro, weighted, micro)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    
    print(f"F1 (Macro):    {f1_macro:.4f}  [penalizes minority classes equally]")
    print(f"F1 (Weighted): {f1_weighted:.4f} [considers class distribution]")
    print(f"F1 (Micro):    {f1_micro:.4f}   [same as accuracy for multiclass]")
    
    # 3. Precision & Recall
    prec_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    
    print(f"\nPrecision (Macro): {prec_macro:.4f}")
    print(f"Recall (Macro):    {recall_macro:.4f}")
    
    # 4. Per-class metrics
    print("\n" + "="*60)
    print("Per-Class Metrics (Top 10 Classes):")
    print("="*60)
    report = classification_report(y_true, y_pred, digits=3, output_dict=False)
    print(report)
    
    # 5. Confusion Matrix (for analysis)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_micro': f1_micro,
        'precision': prec_macro,
        'recall': recall_macro,
        'confusion_matrix': cm
    }
```

### Which Metric to Prioritize?

| Metric | When to Use |
|--------|------------|
| **Macro F1** | If all classes equally important (fair for tail classes) |
| **Weighted F1** | If class distribution reflects real-world deployment |
| **Accuracy** | Good for balanced datasets OR combined with macro F1 |
| **Per-class F1** | Identify weak classes; especially minority classes |

**Recommendation for Your Task**: Monitor **Weighted F1** (primary) + **Macro F1** (secondary) + **Accuracy**.

### Analyzing Confusion Matrix

```python
import numpy as np
import matplotlib.pyplot as plt

def analyze_confusion_matrix(cm, class_names):
    """Find worst-performing classes"""
    
    # Recall per class (diagonal / row sum)
    recalls = cm.diagonal() / cm.sum(axis=1)
    
    # Find worst classes
    worst_idx = np.argsort(recalls)[:5]  # Bottom 5
    
    print("Worst-performing classes:")
    for idx in worst_idx:
        print(f"  {class_names[idx]}: {recalls[idx]:.2%} recall")
    
    # Find most confused pairs
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Get top off-diagonal elements
    for i in range(len(cm)):
        top_confusion = np.argsort(cm_normalized[i, :])[-2]  # 2nd highest (1st is diagonal)
        if top_confusion != i and cm[i, top_confusion] > 5:
            print(f"\n{class_names[i]} often confused with {class_names[top_confusion]}")
            print(f"  Confusion rate: {cm_normalized[i, top_confusion]:.2%}")
```

### Misclassification Analysis (Debug)

```python
def analyze_misclassifications(images, y_true, y_pred, class_names, top_k=10):
    """Visualize hardest examples"""
    
    # Get confidence scores
    y_pred_proba = model(images)  # Or use cached probs
    confidences = y_pred_proba.max(dim=1).values
    
    # Find confident but wrong predictions
    wrong_mask = y_true != y_pred
    wrong_confidence = confidences[wrong_mask]
    wrong_idx = np.argsort(wrong_confidence.numpy())[-top_k:]  # Most confident wrongs
    
    print("Hardest misclassifications (high confidence, wrong label):")
    for idx in wrong_idx:
        true_class = class_names[y_true[idx]]
        pred_class = class_names[y_pred[idx]]
        conf = confidences[idx].item()
        print(f"  True: {true_class}, Pred: {pred_class}, Conf: {conf:.2%}")
        # Display image for manual inspection
```

---

## 9. Avoiding Overfitting: Practical Strategies

### Multi-Layer Defense

1. **Regularization (Weight Decay)**
   - L2 penalty on weights during backprop
   - Implementation: `weight_decay=0.0001` in optimizer

2. **Data Augmentation** (Section 7)
   - Increases effective dataset diversity
   - Reduces model's ability to memorize

3. **Dropout** (Already in MobileNetV3/EfficientNet)
   - Randomly zeroes activations during training
   - Pre-trained models include this

4. **Early Stopping** (Section 5)
   - Stop training when validation loss plateaus
   - Prevents further overfitting

5. **Batch Normalization**
   - Stabilizes training, acts as regularizer
   - Pre-trained models include this

### Monitoring Overfitting

```python
def monitor_overfitting(train_loss, val_loss, epoch, threshold=0.15):
    """Alert if overfitting detected"""
    
    if len(train_loss) > 10:
        gap = val_loss[-1] - train_loss[-1]
        
        if gap > threshold:
            print(f"âš ï¸  WARNING: High overfitting detected at epoch {epoch}")
            print(f"   Gap: {gap:.3f} (train: {train_loss[-1]:.3f}, val: {val_loss[-1]:.3f})")
            print("   â†’ Consider: early stopping, more augmentation, higher weight_decay")
            return True
    
    return False
```

### If Overfitting Persists: Escalation Plan

| Severity | Action | Expected Improvement |
|----------|--------|---------------------|
| Gap 5â€“10% | Increase weight_decay to 0.0005 | -3% gap |
| Gap 10â€“20% | Add stronger augmentation | -5% gap |
| Gap 20â€“35% | Reduce model complexity (use MobileNetV3-Small) | -10% gap |
| Gap > 35% | Reduce epochs OR collect more data | Fundamental data issue |

---

## 10. Clean Training Implementation Checklist

### Pre-Training Setup

- [ ] **Reproducibility**
  ```python
  def seed_all(seed=42):
      torch.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      np.random.seed(seed)
      random.seed(seed)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
  
  seed_all(42)
  ```

- [ ] **Data Pipeline**
  ```python
  # Verify no data leakage
  assert len(set(train_paths) & set(val_paths)) == 0
  # Check class distribution
  print(f"Train classes: {len(set(train_labels))}, Val classes: {len(set(val_labels))}")
  ```

- [ ] **Model & Loss Function**
  ```python
  model = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=80)
  criterion = nn.CrossEntropyLoss(weight=class_weights)
  optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
  scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
  ```

### Training Loop

- [ ] **Track Metrics**
  ```python
  history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}
  ```

- [ ] **Checkpointing**
  ```python
  best_val_acc = 0
  for epoch in range(max_epochs):
      # ... training ...
      if val_acc > best_val_acc:
          best_val_acc = val_acc
          torch.save(model.state_dict(), 'best_model.pth')
  ```

- [ ] **Gradient Clipping** (Optional, helps with stability)
  ```python
  for epoch in range(max_epochs):
      # ... forward pass, loss ...
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()
  ```

- [ ] **Validation Every Epoch**
  ```python
  with torch.no_grad():
      val_loss, val_acc = validate(model, val_loader)
      history['val_loss'].append(val_loss)
  ```

### Post-Training

- [ ] **Evaluate on Test Set** (separate from validation)
  ```python
  test_acc, test_f1, test_cm = evaluate_model(model, test_loader)
  ```

- [ ] **Save Model + Config**
  ```python
  config = {
      'model': 'mobilenetv3_large_100',
      'num_classes': 80,
      'image_size': 224,
      'batch_size': 64,
      'lr': 0.001,
      'weight_decay': 0.0001,
      'val_acc': test_acc,
      'val_f1': test_f1
  }
  torch.save({
      'model_state': model.state_dict(),
      'config': config
  }, 'final_model.pth')
  ```

- [ ] **Reproducibility Report**
  ```python
  print(f"""
  Training Summary:
  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  Model: {config['model']}
  Final Val Accuracy: {test_acc:.4f}
  Final Weighted F1: {test_f1:.4f}
  Classes: {config['num_classes']}
  Image Size: {config['image_size']}Ã—{config['image_size']}
  """)
  ```

---

## 11. Learning Resources & Implementation Order

### Phase 1: Foundation (Weeks 1â€“2)

**Goal**: Understand core concepts before implementation

1. **Transfer Learning & Fine-tuning**
   - Read: [Keras Transfer Learning Guide](https://keras.io/guides/transfer_learning/)
   - Watch: Fast.ai Lesson 2 (Transfer Learning)
   - Time: 2â€“3 hours
   - **Key Concepts**: Feature extraction, frozen vs unfrozen layers, warmth metaphor

2. **Learning Rate Scheduling**
   - Read: [PyTorch Scheduler Documentation](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
   - Paper: "Cyclical Learning Rates for Training Neural Networks" (Leslie Smith, 2015)
   - Implement: Simple cosine annealing scheduler
   - Time: 1â€“2 hours

3. **Data Augmentation**
   - Read: [Albumentations Documentation](https://albumentations.ai/)
   - Visualize: Augmentation pipeline on sample images
   - Time: 1â€“2 hours

**Implementation Task**: Build a minimal PyTorch training loop (frozen backbone only, 10% dataset)

---

### Phase 2: Diagnosis (Weeks 2â€“3)

**Goal**: Interpret training curves, detect problems

1. **Reading Learning Curves**
   - Read: [Towards Data Science: Underfitting vs Overfitting](https://sourestdeeds.github.io/blog/overfitting-and-underfitting/)
   - Paper: "Using Training History to Detect Overfitting" (OpenReview 2024)
   - Implement: Plot utility + diagnostic function
   - Time: 1â€“2 hours

2. **Handling Class Imbalance**
   - Read: [Focal Loss Paper](https://arxiv.org/abs/1708.02002) (sections 1â€“3 only)
   - Tutorial: [Weighted Loss in PyTorch](https://stackoverflow.com/questions/64751157/)
   - Implement: Both weighted CE and focal loss
   - Time: 2â€“3 hours

3. **Evaluation Metrics**
   - Read: [F1 Score vs Accuracy](https://encord.com/blog/f1-score-in-machine-learning/)
   - Notebook: [Confusion Matrix Tutorial](https://iamirmasoud.com/2022/06/19/)
   - Implement: Comprehensive evaluation function
   - Time: 1â€“2 hours

**Implementation Task**: Train on 10% dataset, diagnose issues, refine hyperparameters

---

### Phase 3: Scaling (Weeks 3â€“4)

**Goal**: Scale training from 10% â†’ 50% â†’ 100%

1. **Reproducibility & Seeds**
   - Read: [PyTorch Reproducibility Guide](https://www.codegenes.net/blog/pytorch-reproducibility/)
   - Implement: seed_all() function, worker_init_fn
   - Time: 0.5â€“1 hour

2. **Early Stopping & Checkpointing**
   - Tutorial: [Managing PyTorch Training](https://machinelearningmastery.com/managing-a-pytorch-training-process-with-checkpoints-and-early-stopping/)
   - Implement: EarlyStopping class + model checkpointing
   - Time: 1â€“2 hours

3. **Layer Unfreezing Strategy**
   - Watch: [Layer Freezing/Unfreezing Concepts](https://www.youtube.com/watch?v=OMtfLgUdWUk)
   - Implement: Gradual unfreezing strategy
   - Time: 1â€“2 hours

**Implementation Task**: Train on 50% dataset, apply unfreezing, validate scaling

---

### Phase 4: Deployment (Week 4â€“5)

**Goal**: Prepare model for mobile deployment

1. **Model Quantization**
   - Read: [TensorFlow Lite Optimization](https://apxml.com/courses/advanced-tensorflow/chapter-6-model-deployment-optimization/optimizing-on-device-inference)
   - Tutorial: [PyTorch to TFLite Quantization](https://deepsense.ai/resource/from-pytorch-to-android-creating-a-quantized-tensorflow-lite-model/)
   - Implement: Post-training quantization
   - Time: 2â€“3 hours

2. **Model Export & Testing**
   - PyTorch â†’ ONNX â†’ TFLite conversion
   - Benchmark on target device (Android/iOS simulator)
   - Time: 1â€“2 hours

**Implementation Task**: Train final model on 100% dataset, quantize, test on mobile

---

### Implementation Order (Do This Sequentially)

```
Week 1:
  Day 1: Learn transfer learning + LR scheduling
  Day 2: Understand augmentation
  Day 3: Build minimal training loop (frozen backbone)
  Day 4â€“5: Train on 10%, diagnose

Week 2:
  Day 1: Learn learning curves + overfitting detection
  Day 2: Implement class imbalance handling
  Day 3: Build evaluation metrics
  Day 4â€“5: Re-train 10% with improvements

Week 3:
  Day 1: Implement reproducibility + early stopping
  Day 2: Implement layer unfreezing
  Day 3â€“4: Train on 50%, validate scaling
  Day 5: Plan 100% training

Week 4:
  Day 1â€“3: Train on 100% dataset
  Day 4â€“5: Quantization + mobile testing
```

---

## Recommended GitHub Repositories & Notebooks

### Reference Implementations

1. **Timm (Pytorch Image Models)**
   - [GitHub: rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
   - Pre-trained MobileNetV3, EfficientNet models
   - Use: `timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=80)`

2. **PyTorch Image Classification**
   - [PyTorch Vision Tutorials](https://pytorch.org/vision/stable/models.html)
   - Official transfer learning guide

3. **Food-Specific Implementations**
   - [MobileNetV2-FoodClassifier](https://github.com/Pramit726/MobileNetV2-FoodClassifier)
   - Real-world food classification example

4. **Albumentations Documentation**
   - [Official Docs](https://albumentations.ai/)
   - Examples for food image augmentation

---

## Summary: Quick Reference Card

| Component | Recommendation | Notes |
|-----------|---|---|
| **Model** | MobileNetV3-Large | Start here; scale to EfficientNet-Lite B2 if needed |
| **Image Size** | 224Ã—224 | Standard for ImageNet pre-trained models |
| **Batch Size** | 64 | Adjust to 32 if OOM |
| **Learning Rate** | 0.001 (frozen), 0.0001 (unfrozen) | Use warmup + cosine annealing |
| **Weight Decay** | 0.0001 | Increase to 0.0005 if overfitting |
| **Epochs** | 30â€“40 | Use early stopping (patience=5) |
| **Loss Function** | Weighted CrossEntropyLoss | Switch to Focal Loss if severe imbalance |
| **Augmentation** | RandAugment + ColorJitter + Dropout | Add incrementally; avoid over-augmentation |
| **Metric to Track** | Weighted F1 + Macro F1 | Better than accuracy for imbalanced data |
| **Dataset Scaling** | 10% â†’ 50% â†’ 100% | Validate each phase before scaling |
| **Unfreezing** | Start frozen, unfreeze after 5 epochs if plateaued | Use lower LR for pre-trained layers |
| **Mobile Deployment** | INT8 quantization + TFLite | Target: <20 MB, <50 ms inference |

---

## Final Notes

1. **Start Simple**: Frozen backbone + weighted loss + basic augmentation often gets 80%+ accuracy
2. **Iterate Methodically**: 10% â†’ 50% â†’ 100%, not just 0% â†’ 100%
3. **Monitor Everything**: Plot curves, track F1, analyze confusion matrices
4. **Version Your Work**: Save configs, checkpoints, hyperparams for reproducibility
5. **Test on Target Device**: Mobile inference speed != desktop latency
6. **Document Trade-offs**: Accuracy vs model size vs inference latency

Good luck! ðŸš€
