"""
RT-DETRv3 PyTorch Configuration System
Adapted from RT-DETRv2 PyTorch implementation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

import os
import copy
import yaml
import re
from pathlib import Path
from typing import Any, Dict, Optional, List, Callable
from dataclasses import dataclass

from .workspace import create, GLOBAL_CONFIG

__all__ = ['RTDETRConfig', 'load_config', 'merge_config', 'merge_dict']

INCLUDE_KEY = '__include__'


@dataclass
class RTDETRConfig:
    """RT-DETRv3 Configuration Class"""
    
    # Model configuration
    backbone: str = "ResNet50"
    neck: str = "HybridEncoder"
    head: str = "RTDETRHead"
    num_classes: int = 80
    hidden_dim: int = 256
    num_queries: int = 300
    num_decoder_layers: int = 6
    
    # Training configuration
    batch_size: int = 16
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    epoches: int = 72
    
    # Data configuration
    input_size: tuple = (640, 640)
    num_workers: int = 4
    
    # Runtime configuration
    device: str = "cuda"
    output_dir: str = "./outputs"
    resume: str = None
    use_amp: bool = False
    use_ema: bool = False
    clip_max_norm: float = 0.1
    
    # Private attributes
    _model: nn.Module = None
    _postprocessor: nn.Module = None
    _criterion: nn.Module = None
    _optimizer: optim.Optimizer = None
    _lr_scheduler: LRScheduler = None
    _train_dataloader: DataLoader = None
    _val_dataloader: DataLoader = None
    _ema: nn.Module = None
    _scaler: GradScaler = None
    _evaluator: Callable = None
    _writer: SummaryWriter = None
    
    def __post_init__(self):
        """Initialize configuration after dataclass creation"""
        if isinstance(self.input_size, list):
            self.input_size = tuple(self.input_size)

    @property
    def model(self) -> nn.Module:
        return self._model

    @model.setter
    def model(self, m):
        assert isinstance(m, nn.Module), f'{type(m)} != nn.Module'
        self._model = m

    @property
    def postprocessor(self) -> nn.Module:
        return self._postprocessor

    @postprocessor.setter
    def postprocessor(self, m):
        assert isinstance(m, nn.Module), f'{type(m)} != nn.Module'
        self._postprocessor = m

    @property
    def criterion(self) -> nn.Module:
        return self._criterion

    @criterion.setter
    def criterion(self, m):
        assert isinstance(m, nn.Module), f'{type(m)} != nn.Module'
        self._criterion = m

    @property
    def optimizer(self) -> optim.Optimizer:
        return self._optimizer

    @optimizer.setter
    def optimizer(self, m):
        assert isinstance(m, optim.Optimizer), f'{type(m)} != optim.Optimizer'
        self._optimizer = m

    @property
    def lr_scheduler(self) -> LRScheduler:
        return self._lr_scheduler

    @lr_scheduler.setter
    def lr_scheduler(self, m):
        assert isinstance(m, LRScheduler), f'{type(m)} != LRScheduler'
        self._lr_scheduler = m

    @property
    def train_dataloader(self) -> DataLoader:
        return self._train_dataloader

    @train_dataloader.setter
    def train_dataloader(self, loader):
        self._train_dataloader = loader

    @property
    def val_dataloader(self) -> DataLoader:
        return self._val_dataloader

    @val_dataloader.setter
    def val_dataloader(self, loader):
        self._val_dataloader = loader

    @property
    def ema(self) -> nn.Module:
        return self._ema

    @ema.setter
    def ema(self, obj):
        self._ema = obj

    @property
    def scaler(self) -> GradScaler:
        if self._scaler is None and self.use_amp and torch.cuda.is_available():
            self._scaler = GradScaler()
        return self._scaler

    @scaler.setter
    def scaler(self, obj: GradScaler):
        self._scaler = obj

    @property
    def evaluator(self) -> Callable:
        return self._evaluator

    @evaluator.setter
    def evaluator(self, fn):
        assert isinstance(fn, Callable), f'{type(fn)} must be Callable'
        self._evaluator = fn

    @property
    def writer(self) -> SummaryWriter:
        if self._writer is None and self.output_dir:
            self._writer = SummaryWriter(Path(self.output_dir) / 'summary')
        return self._writer

    @writer.setter
    def writer(self, m):
        assert isinstance(m, SummaryWriter), f'{type(m)} must be SummaryWriter'
        self._writer = m


class YAMLConfig(RTDETRConfig):
    """YAML-based Configuration for RT-DETRv3"""
    
    def __init__(self, cfg_path: str, **kwargs):
        # Load YAML config first
        cfg = load_config(cfg_path)
        cfg = merge_dict(cfg, kwargs)
        
        # Store the raw config
        self.yaml_cfg = copy.deepcopy(cfg)
        
        # Initialize dataclass with defaults
        super().__init__()
        
        # Override with YAML values
        for key, value in cfg.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    @property
    def global_cfg(self):
        return merge_config(self.yaml_cfg, inplace=False, overwrite=False)
    
    @property
    def model(self) -> nn.Module:
        if self._model is None and 'model' in self.yaml_cfg:
            self._model = create(self.yaml_cfg['model'], self.global_cfg)
        return self._model
    
    @property
    def postprocessor(self) -> nn.Module:
        if self._postprocessor is None and 'postprocessor' in self.yaml_cfg:
            self._postprocessor = create(self.yaml_cfg['postprocessor'], self.global_cfg)
        return self._postprocessor
    
    @property
    def criterion(self) -> nn.Module:
        if self._criterion is None and 'criterion' in self.yaml_cfg:
            self._criterion = create(self.yaml_cfg['criterion'], self.global_cfg)
        return self._criterion
    
    @property
    def optimizer(self) -> optim.Optimizer:
        if self._optimizer is None and 'optimizer' in self.yaml_cfg:
            params = self.get_optim_params(self.yaml_cfg['optimizer'], self.model)
            self._optimizer = create('optimizer', self.global_cfg, params=params)
        return self._optimizer
    
    @property
    def lr_scheduler(self) -> LRScheduler:
        if self._lr_scheduler is None and 'lr_scheduler' in self.yaml_cfg:
            self._lr_scheduler = create('lr_scheduler', self.global_cfg, optimizer=self.optimizer)
        return self._lr_scheduler
    
    @property
    def train_dataloader(self) -> DataLoader:
        if self._train_dataloader is None and 'train_dataloader' in self.yaml_cfg:
            self._train_dataloader = self.build_dataloader('train_dataloader')
        return self._train_dataloader
    
    @property
    def val_dataloader(self) -> DataLoader:
        if self._val_dataloader is None and 'val_dataloader' in self.yaml_cfg:
            self._val_dataloader = self.build_dataloader('val_dataloader')
        return self._val_dataloader
    
    @property
    def ema(self) -> nn.Module:
        if self._ema is None and self.use_ema:
            from ..optim import ModelEMA
            self._ema = ModelEMA(self.model, decay=0.9999, warmups=2000)
        return self._ema
    
    @property
    def evaluator(self):
        if self._evaluator is None and 'evaluator' in self.yaml_cfg:
            if self.yaml_cfg['evaluator']['type'] == 'CocoEvaluator':
                from ..data import get_coco_api_from_dataset
                base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
                self._evaluator = create('evaluator', self.global_cfg, coco_gt=base_ds)
            else:
                raise NotImplementedError(f"{self.yaml_cfg['evaluator']['type']}")
        return self._evaluator
    
    @staticmethod
    def get_optim_params(cfg: dict, model: nn.Module):
        """Get optimizer parameters with regex pattern matching"""
        assert 'type' in cfg
        cfg = copy.deepcopy(cfg)
        
        if 'params' not in cfg:
            return model.parameters()
        
        assert isinstance(cfg['params'], list)
        
        param_groups = []
        visited = []
        
        for pg in cfg['params']:
            pattern = pg['params']
            params = {k: v for k, v in model.named_parameters() 
                     if v.requires_grad and len(re.findall(pattern, k)) > 0}
            pg['params'] = params.values()
            param_groups.append(pg)
            visited.extend(list(params.keys()))
        
        # Add remaining parameters
        names = [k for k, v in model.named_parameters() if v.requires_grad]
        if len(visited) < len(names):
            unseen = set(names) - set(visited)
            params = {k: v for k, v in model.named_parameters() 
                     if v.requires_grad and k in unseen}
            param_groups.append({'params': params.values()})
        
        return param_groups
    
    def build_dataloader(self, name: str):
        """Build dataloader from configuration"""
        from ..misc import dist_utils
        
        # Handle distributed training batch size
        total_batch_size = self.yaml_cfg[name].get('total_batch_size', None)
        if total_batch_size is not None:
            world_size = dist_utils.get_world_size() if hasattr(dist_utils, 'get_world_size') else 1
            batch_size = total_batch_size // world_size
        else:
            batch_size = self.yaml_cfg[name].get('batch_size', self.batch_size)
        
        print(f'Building {name} with batch_size={batch_size}...')
        
        global_cfg = self.global_cfg
        if 'total_batch_size' in global_cfg[name]:
            _ = global_cfg[name].pop('total_batch_size')
        
        loader = create(name, global_cfg, batch_size=batch_size)
        loader.shuffle = self.yaml_cfg[name].get('shuffle', name == 'train_dataloader')
        
        return loader


def load_config(file_path: str, cfg: dict = None) -> dict:
    """Load configuration from YAML file with include support"""
    if cfg is None:
        cfg = {}
    
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "Only support YAML files"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        file_cfg = yaml.safe_load(f)
        if file_cfg is None:
            return {}
    
    # Handle includes
    if INCLUDE_KEY in file_cfg:
        base_yamls = list(file_cfg[INCLUDE_KEY])
        for base_yaml in base_yamls:
            if base_yaml.startswith('~'):
                base_yaml = os.path.expanduser(base_yaml)
            
            if not base_yaml.startswith('/'):
                base_yaml = os.path.join(os.path.dirname(file_path), base_yaml)
            
            base_cfg = load_config(base_yaml, cfg)
            merge_dict(cfg, base_cfg)
    
    return merge_dict(cfg, file_cfg)


def merge_dict(dct: dict, another_dct: dict, inplace: bool = True) -> dict:
    """Merge another_dct into dct"""
    def _merge(dct, another):
        for k in another:
            if (k in dct and isinstance(dct[k], dict) and isinstance(another[k], dict)):
                _merge(dct[k], another[k])
            else:
                dct[k] = another[k]
        return dct
    
    if not inplace:
        dct = copy.deepcopy(dct)
    
    return _merge(dct, another_dct)


def merge_config(cfg: dict, another_cfg: dict = None, inplace: bool = False, overwrite: bool = False) -> dict:
    """Merge configuration with GLOBAL_CONFIG"""
    if another_cfg is None:
        another_cfg = GLOBAL_CONFIG
    
    def _merge(dct, another):
        for k in another:
            if k not in dct:
                dct[k] = another[k]
            elif isinstance(dct[k], dict) and isinstance(another[k], dict):
                _merge(dct[k], another[k])
            elif overwrite:
                dct[k] = another[k]
        return dct
    
    if not inplace:
        cfg = copy.deepcopy(cfg)
    
    return _merge(cfg, another_cfg)


def parse_cli(nargs: List[str]) -> Dict:
    """Parse command-line arguments"""
    def dictify(s: str, v: Any) -> Dict:
        if '.' not in s:
            return {s: v}
        key, rest = s.split('.', 1)
        return {key: dictify(rest, v)}
    
    cfg = {}
    if nargs is None or len(nargs) == 0:
        return cfg
    
    for s in nargs:
        s = s.strip()
        k, v = s.split('=', 1)
        d = dictify(k, yaml.safe_load(v))
        cfg = merge_dict(cfg, d)
    
    return cfg