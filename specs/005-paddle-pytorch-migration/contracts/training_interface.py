"""
Training Interface Contract: RT-DETRv3 PyTorch Migration

Defines the interface contracts for training engine, optimizer, scheduler, and evaluator.
"""

from typing import Dict, List, Optional, Callable, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader


class TrainerInterface:
    """
    Interface contract for training engine.

    Manages the training loop, optimization, and evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        lr_scheduler: _LRScheduler,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        max_epochs: int = 72,
        save_dir: str = 'output/',
        log_iter: int = 10,
        eval_epoch: int = 1,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        use_amp: bool = False,
        gradient_clip: Optional[float] = None,
        ema_decay: Optional[float] = None
    ):
        """
        Initialize Trainer.

        Args:
            model: Detection model
            optimizer: Optimizer instance
            lr_scheduler: Learning rate scheduler
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            max_epochs: Maximum training epochs
            save_dir: Directory to save checkpoints
            log_iter: Log every N iterations
            eval_epoch: Evaluate every N epochs
            device: Training device
            use_amp: Whether to use automatic mixed precision
            gradient_clip: Gradient clipping value (None to disable)
            ema_decay: EMA decay rate (None to disable)
        """
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.save_dir = save_dir
        self.log_iter = log_iter
        self.eval_epoch = eval_epoch
        self.device = device
        self.use_amp = use_amp
        self.gradient_clip = gradient_clip
        self.ema_decay = ema_decay

    def train(self) -> None:
        """
        Main training loop.

        Iterates over epochs, calls train_epoch() and evaluate() as needed.
        """
        raise NotImplementedError

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dict of average losses:
            {
                'loss': total loss,
                'loss_class': classification loss,
                'loss_bbox': bbox loss,
                'loss_giou': giou loss
            }
        """
        raise NotImplementedError

    def evaluate(self, epoch: int) -> Dict[str, float]:
        """
        Evaluate on validation set.

        Args:
            epoch: Current epoch number

        Returns:
            Dict of metrics:
            {
                'mAP': mean Average Precision,
                'AP50': AP at IoU=0.50,
                'AP75': AP at IoU=0.75,
                ...
            }
        """
        raise NotImplementedError

    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        filename: Optional[str] = None
    ) -> None:
        """
        Save checkpoint.

        Args:
            epoch: Current epoch
            is_best: Whether this is the best checkpoint so far
            filename: Custom filename (default: 'checkpoint_epoch_{epoch}.pth')
        """
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_path: str, resume: bool = True) -> int:
        """
        Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            resume: If True, load optimizer and scheduler states

        Returns:
            Starting epoch number
        """
        raise NotImplementedError


class OptimizerBuilderInterface:
    """
    Interface for building optimizers.

    Supports different optimizer types and learning rate strategies.
    """

    @staticmethod
    def build(
        parameters: Any,
        optimizer_type: str = 'AdamW',
        lr: float = 0.0001,
        weight_decay: float = 0.0001,
        **kwargs
    ) -> optim.Optimizer:
        """
        Build optimizer.

        Args:
            parameters: Model parameters
            optimizer_type: Optimizer name ('AdamW', 'SGD', 'Adam')
            lr: Learning rate
            weight_decay: Weight decay coefficient
            **kwargs: Additional optimizer-specific arguments

        Returns:
            Optimizer instance
        """
        if optimizer_type == 'AdamW':
            return optim.AdamW(
                parameters,
                lr=lr,
                weight_decay=weight_decay,
                betas=kwargs.get('betas', (0.9, 0.999)),
                eps=kwargs.get('eps', 1e-8)
            )
        elif optimizer_type == 'SGD':
            return optim.SGD(
                parameters,
                lr=lr,
                momentum=kwargs.get('momentum', 0.9),
                weight_decay=weight_decay
            )
        elif optimizer_type == 'Adam':
            return optim.Adam(
                parameters,
                lr=lr,
                weight_decay=weight_decay,
                betas=kwargs.get('betas', (0.9, 0.999)),
                eps=kwargs.get('eps', 1e-8)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")


class LRSchedulerBuilderInterface:
    """
    Interface for building learning rate schedulers.

    Supports common scheduler types and composition (warmup + main scheduler).
    """

    @staticmethod
    def build(
        optimizer: optim.Optimizer,
        scheduler_type: str = 'CosineAnnealing',
        max_epochs: int = 72,
        warmup_steps: int = 1000,
        warmup_start_lr: float = 0.0,
        **kwargs
    ) -> _LRScheduler:
        """
        Build learning rate scheduler.

        Args:
            optimizer: Optimizer instance
            scheduler_type: Scheduler name ('CosineAnnealing', 'MultiStep', 'Linear')
            max_epochs: Total training epochs
            warmup_steps: Warmup iterations
            warmup_start_lr: Initial learning rate for warmup
            **kwargs: Additional scheduler-specific arguments

        Returns:
            Scheduler instance (possibly composite with warmup)
        """
        from torch.optim.lr_scheduler import (
            CosineAnnealingLR, MultiStepLR, LinearLR, SequentialLR
        )

        base_lr = optimizer.param_groups[0]['lr']

        # Build warmup scheduler
        if warmup_steps > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=warmup_start_lr / base_lr,
                end_factor=1.0,
                total_iters=warmup_steps
            )
        else:
            warmup_scheduler = None

        # Build main scheduler
        if scheduler_type == 'CosineAnnealing':
            main_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=kwargs.get('T_max', max_epochs * 1000),  # Assume ~1000 iters/epoch
                eta_min=kwargs.get('eta_min', 0.0)
            )
        elif scheduler_type == 'MultiStep':
            main_scheduler = MultiStepLR(
                optimizer,
                milestones=kwargs.get('milestones', [40, 60]),
                gamma=kwargs.get('gamma', 0.1)
            )
        elif scheduler_type == 'Linear':
            main_scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=kwargs.get('end_factor', 0.1),
                total_iters=kwargs.get('total_iters', max_epochs * 1000)
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")

        # Compose warmup + main scheduler
        if warmup_scheduler is not None:
            return SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_steps]
            )
        else:
            return main_scheduler


class EMAInterface:
    """
    Interface for Exponential Moving Average of model weights.

    Maintains a shadow copy of model parameters for more stable inference.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: Optional[torch.device] = None
    ):
        """
        Initialize EMA.

        Args:
            model: Model to track
            decay: EMA decay rate
            device: Device to store EMA weights
        """
        self.model = model
        self.decay = decay
        self.device = device if device is not None else next(model.parameters()).device
        self.shadow = {}
        self.backup = {}

    def register(self) -> None:
        """Register model parameters for EMA tracking."""
        raise NotImplementedError

    def update(self) -> None:
        """Update EMA weights: shadow = decay * shadow + (1 - decay) * param."""
        raise NotImplementedError

    def apply_shadow(self) -> None:
        """Replace model parameters with EMA shadow weights (for evaluation)."""
        raise NotImplementedError

    def restore(self) -> None:
        """Restore original model parameters (after evaluation)."""
        raise NotImplementedError


class EvaluatorInterface:
    """
    Interface for model evaluation.

    Computes COCO-style metrics (mAP, AP50, AP75, etc.).
    """

    def __init__(
        self,
        dataset_dir: str,
        anno_file: str,
        num_classes: int = 80
    ):
        """
        Initialize evaluator.

        Args:
            dataset_dir: Dataset root directory
            anno_file: Annotation file path (COCO format JSON)
            num_classes: Number of classes
        """
        self.dataset_dir = dataset_dir
        self.anno_file = anno_file
        self.num_classes = num_classes

    def evaluate(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: torch.device
    ) -> Dict[str, float]:
        """
        Evaluate model on dataset.

        Args:
            model: Detection model
            data_loader: Data loader
            device: Evaluation device

        Returns:
            Dict of metrics:
            {
                'mAP': float,
                'AP50': float,
                'AP75': float,
                'APs': float (small objects),
                'APm': float (medium objects),
                'APl': float (large objects)
            }
        """
        raise NotImplementedError

    def process_predictions(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        image_ids: List[int]
    ) -> List[Dict]:
        """
        Convert model predictions to COCO format.

        Args:
            predictions: List of prediction dicts from model
            image_ids: Corresponding image IDs

        Returns:
            List of COCO-format detection dicts:
            [
                {
                    'image_id': int,
                    'category_id': int,
                    'bbox': [x, y, w, h],
                    'score': float
                },
                ...
            ]
        """
        raise NotImplementedError


class CheckpointManagerInterface:
    """
    Interface for checkpoint management.

    Handles saving/loading checkpoints and keeping best N checkpoints.
    """

    def __init__(
        self,
        save_dir: str,
        max_checkpoints: int = 5,
        best_metric: str = 'mAP'
    ):
        """
        Initialize checkpoint manager.

        Args:
            save_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            best_metric: Metric name for determining best checkpoint
        """
        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.best_metric = best_metric
        self.best_value = 0.0
        self.checkpoints = []

    def save(
        self,
        state_dict: Dict[str, Any],
        epoch: int,
        metric_value: Optional[float] = None,
        is_best: bool = False
    ) -> str:
        """
        Save checkpoint.

        Args:
            state_dict: Checkpoint state dict
            epoch: Current epoch
            metric_value: Current metric value (for best checkpoint tracking)
            is_best: Whether to save as 'best.pth'

        Returns:
            Path to saved checkpoint
        """
        raise NotImplementedError

    def load(
        self,
        checkpoint_path: str,
        map_location: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            map_location: Device to map tensors to

        Returns:
            Checkpoint state dict
        """
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        return checkpoint


# Type aliases for clarity
StateDict = Dict[str, Any]
MetricsDict = Dict[str, float]
LossDict = Dict[str, float]
