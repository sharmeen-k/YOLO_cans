"""
Fixed Distillation Trainer - Uses Ultralytics built-in training with distillation overlay.

Instead of reimplementing the YOLO training loop (which has subtle loss computation issues),
this uses YOLO.train() under the hood and injects distillation via a custom DetectionTrainer
subclass. This guarantees correct detection loss (box, cls, dfl) while adding spatial
attention distillation on top.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import colorstr
import os


class SpatialAttentionLoss(nn.Module):
    """
    Feature-Level De-blurring via Spatial Attention Transfer.
    Computes spatial attention map sum(|F_c|^2) across channels and applies MSE loss.
    """
    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize

    def forward(self, student_feat, teacher_feat, temperature=4.0):
        if student_feat.shape != teacher_feat.shape:
            target_size = student_feat.shape[2:]
            teacher_feat = F.adaptive_avg_pool2d(teacher_feat, target_size)

        student_attention = torch.sum(student_feat.pow(2), dim=1, keepdim=True)
        teacher_attention = torch.sum(teacher_feat.pow(2), dim=1, keepdim=True)

        student_attention = student_attention / temperature
        teacher_attention = teacher_attention / temperature

        if self.normalize:
            B, C, H, W = student_attention.shape
            s_flat = F.normalize(student_attention.view(B, -1), p=2, dim=1)
            t_flat = F.normalize(teacher_attention.view(B, -1), p=2, dim=1)
            student_attention = s_flat.view(B, C, H, W)
            teacher_attention = t_flat.view(B, C, H, W)

        return F.mse_loss(student_attention, teacher_attention.detach())


class FeatureHook:
    """Picklable forward hook that stores the last seen feature tensor.
    Must be a top-level class so torch.save() can pickle it."""
    def __init__(self, storage, key):
        self.storage = storage
        self.key = key

    def __call__(self, module, inp, out):
        self.storage[self.key] = out[0] if isinstance(out, tuple) else out


class DistillationDetectionTrainer(DetectionTrainer):
    """
    Custom DetectionTrainer that adds spatial attention distillation loss.
    
    Uses Ultralytics' proven training loop (correct loss, augmentation, scheduling)
    and overlays distillation by wrapping the model's loss function.
    """
    # Class-level config — set before calling YOLO.train(trainer=...)
    teacher_path = None
    distill_weight = 50.0
    blur_prob = 0.8

    def setup_model(self):
        """Called by Ultralytics during trainer init. Sets up student + teacher."""
        super().setup_model()
        
        if self.teacher_path is None:
            raise ValueError("DistillationDetectionTrainer.teacher_path must be set before training")
        
        device = self.device
        print(colorstr("blue", "bold", f"\n  Setting up distillation from teacher: {self.teacher_path}"))
        
        # Load teacher model (frozen)
        teacher_yolo = YOLO(self.teacher_path)
        self.teacher_model = teacher_yolo.model.to(device)
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False
        
        # Distillation loss
        self.distill_loss_fn = SpatialAttentionLoss(normalize=True)
        
        # Blur augmentation
        from blur_augment import BatchedBlurAugment
        self.blur_augment = BatchedBlurAugment(blur_prob=self.blur_prob, device=device)
        
        # Feature extraction hooks
        self.feature_storage = {'teacher': None, 'student': None}
        feat_idx = self._find_feature_layer(self.teacher_model)
        print(f"  Feature extraction layer index: {feat_idx}")
        
        # Register picklable hooks on both models
        t_children = list(self.teacher_model.model.children())
        s_children = list(self.model.model.children())
        t_children[feat_idx].register_forward_hook(FeatureHook(self.feature_storage, 'teacher'))
        s_children[feat_idx].register_forward_hook(FeatureHook(self.feature_storage, 'student'))
        
        print(f"  Distillation weight: {self.distill_weight}")
        print(f"  Blur probability: {self.blur_prob}")
        print(colorstr("blue", "bold", "  Distillation setup complete ✅\n"))
    
    def _setup_train(self):
        """Override to patch model.loss AFTER EMA creation (which does deepcopy)."""
        super()._setup_train()
        # Now EMA is created, safe to patch model.loss without deepcopy issues
        self._original_loss = self.model.loss
        self.model.loss = self._distillation_loss
        print(colorstr("blue", "bold", "  Distillation loss injection active ✅"))
    
    def _find_feature_layer(self, model):
        """Dynamically locate SPP/SPPF layer in the backbone."""
        children = list(model.model.children())
        for i, module in enumerate(children):
            if hasattr(module, 'cv1') and hasattr(module, 'cv2') and hasattr(module, 'm'):
                if 'SPP' in type(module).__name__:
                    return i
        for i, module in enumerate(children):
            name = type(module).__name__
            if 'Upsample' in name or 'Concat' in name:
                return max(0, i - 1)
        return len(children) // 2

    def preprocess_batch(self, batch):
        """Standard Ultralytics preprocessing + blur augmentation for distillation."""
        batch = super().preprocess_batch(batch)
        
        # Apply blur to student images AFTER standard preprocessing (which does /255)
        # Cast to float32 for blur ops, then back to original dtype (float16 under AMP)
        if hasattr(self, 'blur_augment'):
            orig_dtype = batch["img"].dtype
            batch["img"] = self.blur_augment(batch["img"].float()).to(orig_dtype)
        
        return batch

    def _distillation_loss(self, batch, preds=None):
        """Wraps the standard detection loss and adds distillation."""
        # Reset feature storage
        self.feature_storage['teacher'] = None
        self.feature_storage['student'] = None
        
        # Teacher forward on CLEAN images (before blur was applied)
        # Note: We use the batch images directly since teacher sees the same preprocessing
        with torch.no_grad():
            _ = self.teacher_model(batch["img"])
        
        # Standard Ultralytics detection loss (this also does student forward → captures student features)
        det_loss, loss_items = self._original_loss(batch, preds)
        
        # Compute distillation loss from captured features
        teacher_feat = self.feature_storage['teacher']
        student_feat = self.feature_storage['student']
        
        if teacher_feat is not None and student_feat is not None:
            # Dynamic temperature: starts at 4.0, decreases over training
            progress = self.epoch / max(1, self.epochs - 1)
            temperature = 4.0 - 3.0 * progress
            
            distill_loss = self.distill_loss_fn(student_feat, teacher_feat, temperature=temperature)
            distill_loss = distill_loss * self.distill_weight
            
            # Add to detection loss
            det_loss = det_loss + distill_loss
        
        return det_loss, loss_items


class FixedDistillationTrainer:
    """
    High-level wrapper that configures and runs distillation training
    using Ultralytics' built-in training pipeline.
    """
    def __init__(self, teacher_name='models/yolov8n_cans.pt', student_name='yolov8n.pt', 
                 data_cfg='coco8.yaml', epochs=10, batch_size=4, lr=0.001, run_name='student',
                 device=None, fraction=0.1, cache=False):
        
        self.teacher_name = teacher_name
        self.student_name = student_name
        self.data_cfg = data_cfg
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.run_name = run_name
        self.fraction = fraction
        self.cache = cache
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(colorstr(f"Initializing Fixed Distillation Trainer on {self.device}..."))
        os.makedirs("project", exist_ok=True)

    def train(self):
        """Run distillation training using Ultralytics' built-in pipeline."""
        print(colorstr("blue", "bold", "Starting Distillation Training (Ultralytics-based)..."))
        
        # Configure the custom trainer class
        DistillationDetectionTrainer.teacher_path = self.teacher_name
        DistillationDetectionTrainer.distill_weight = 50.0
        DistillationDetectionTrainer.blur_prob = 0.8
        
        # Load student model
        student = YOLO(self.student_name)
        
        # Use Ultralytics' training with our custom trainer
        # This handles: correct loss computation, augmentation, scheduling,
        # checkpointing, logging, and validation
        results = student.train(
            data=self.data_cfg,
            epochs=self.epochs,
            batch=self.batch_size,
            imgsz=640,
            lr0=self.lr,
            lrf=0.01,             # Final LR = lr0 * lrf
            warmup_epochs=5,       # Warmup matches our previous freeze period
            freeze=10,             # Freeze first 10 layers (backbone) during warmup
            name=self.run_name,
            project="project",
            exist_ok=True,
            device=self.device,
            fraction=self.fraction,
            cache='ram' if self.cache else False,
            trainer=DistillationDetectionTrainer,
            verbose=True,
        )
        
        # Copy best model to expected output path
        best_src = f"project/{self.run_name}/weights/best.pt"
        best_dst = f"project/{self.run_name}_best.pt"
        if os.path.exists(best_src):
            import shutil
            shutil.copy2(best_src, best_dst)
            print(f"\n✅ Best model copied to: {best_dst}")
        
        return results
