"""
Optimizer configurations for RT-DETRv3
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any
from ..core.workspace import register


class ModelEMA:
    """Model Exponential Moving Average"""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999, warmups: int = 2000):
        self.model = model
        self.decay = decay
        self.warmups = warmups
        self.updates = 0
        
        # Create EMA model
        self.ema_model = type(model)(model.config) if hasattr(model, 'config') else None
        if self.ema_model is not None:
            self.ema_model.load_state_dict(model.state_dict())
            self.ema_model.eval()
        
        # Disable gradients for EMA model
        for param in self.ema_model.parameters():
            param.requires_grad = False
    
    def update(self, model: nn.Module):
        """Update EMA model"""
        with torch.no_grad():
            self.updates += 1
            decay = self.decay if self.updates > self.warmups else 1.0
            
            for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.mul_(decay).add_(model_param, alpha=1 - decay)
    
    def __call__(self, model: nn.Module):
        """Update EMA model (callable interface)"""
        self.update(model)
    
    def state_dict(self):
        """Get EMA model state dict"""
        return self.ema_model.state_dict()
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load EMA model state dict"""
        self.ema_model.load_state_dict(state_dict)


@register()
class AdamW:
    """AdamW optimizer configuration"""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
    
    def __call__(self, params=None):
        if params is None:
            params = self.params
        return torch.optim.AdamW(
            params,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay
        )


@register()
class SGD:
    """SGD optimizer configuration"""
    
    def __init__(self, params, lr=1e-3, momentum=0.9, dampening=0, weight_decay=0, nesterov=False):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
    
    def __call__(self, params=None):
        if params is None:
            params = self.params
        return torch.optim.SGD(
            params,
            lr=self.lr,
            momentum=self.momentum,
            dampening=self.dampening,
            weight_decay=self.weight_decay,
            nesterov=self.nesterov
        )