# Copyright (c) 2025 RT-DETRv3 PyTorch Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from PaddlePaddle RT-DETRv3
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

"""
Trainer for RT-DETRv3 PyTorch.

Configuration-driven trainer compatible with Paddle's Trainer API.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import yaml
from collections.abc import Mapping
from typing import Dict, Optional
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler

from .. import modeling as _modeling  # noqa: F401 - trigger component registration
from ..core.config.schema import SchemaDict
from ..core.workspace import create
from ..utils.logger import setup_logger
from ..utils.checkpoint import load_weight, load_pretrain_weight, convert_to_dict
from ..utils.visualizer import visualize_results, save_result
from ..metrics import get_infer_results
from ..data.source.category import get_categories
from ..optimizer import ModelEMA
from .callbacks import ComposeCallback, LogPrinter, Checkpointer
from .naive_sync_bn import convert_syncbn

MOT_ARCH = ['JDE', 'FairMOT', 'DeepSORT', 'ByteTrack', 'CenterTrack']
logger = setup_logger('rtdetrv3.engine')

__all__ = ['Trainer']


class Trainer:
    """
    Configuration-driven Trainer for RT-DETRv3.

    Compatible with Paddle's Trainer API and initialization pattern.

    Args:
        cfg: Configuration object with all training settings
        mode: Training mode ('train', 'eval', or 'test')
    """

    def __init__(self, cfg, mode='train'):
        """
        Initialize trainer from configuration.

        This follows Paddle's initialization pattern:
        1. Parse cfg and setup mode
        2. Build dataset and dataloader using create()
        3. Build model using create()
        4. Build optimizer and scheduler using create()
        5. Setup AMP, EMA, callbacks, metrics

        Args:
            cfg: Configuration object (from ppdet_pytorch.core.workspace)
            mode: 'train', 'eval', or 'test'
        """
        self.cfg = cfg.copy()
        for key, value in cfg.items():
            if isinstance(value, SchemaDict):
                self.cfg[key] = deepcopy(dict(value))
            else:
                self.cfg[key] = deepcopy(value)
        assert mode.lower() in ['train', 'eval', 'test'], \
            "mode should be 'train', 'eval' or 'test'"
        self.mode = mode.lower()

        # Training flags
        self.optimizer = None
        self.is_loaded_weights = False

        # AMP settings (Paddle compatible)
        self.use_amp = self.cfg.get('amp', False)
        self.amp_level = self.cfg.get('amp_level', 'O1')  # Only O1 in PyTorch

        # Distributed training settings
        log_ranks = cfg.get('log_ranks', '0')
        if isinstance(log_ranks, str):
            self.log_ranks = [int(i) for i in log_ranks.split(',')]
        elif isinstance(log_ranks, int):
            self.log_ranks = [log_ranks]
        else:
            self.log_ranks = [0]

        # Create save directory
        # TODO need clarify 
        train_results_path = os.path.abspath(
            os.path.join(self.cfg.save_dir, "train_result.json"))
        self.save_dir = cfg.get('save_dir', './output')
        if dist.is_initialized():
            if dist.get_rank() == 0:
                os.makedirs(self.save_dir, exist_ok=True)
        else:
            os.makedirs(self.save_dir, exist_ok=True)

        # Save config to output directory
        if (not dist.is_initialized() or dist.get_rank() == 0):
            config_path = os.path.join(self.save_dir, "config.yaml")
            if not os.path.exists(config_path):
                with open(config_path, "w") as f:
                    # Convert cfg to dict for saving
                    config_dict = self._convert_cfg_to_dict(cfg)
                    yaml.dump(config_dict, f)

        # Build dataset and dataloader (using create() factory)
        self._build_data(self.cfg)

        # Build model (using create() factory)
        self._build_model(self.cfg)

        # Build optimizer and scheduler (only in train mode)
        if self.mode == 'train':
            self._build_optimizer(self.cfg)

        # Setup AMP
        self.scaler = GradScaler() if self.use_amp else None

        # Setup SyncBatchNorm for distributed training
        if dist.is_initialized() and dist.get_world_size() > 1:
            norm_type = cfg.get('norm_type', None)
            if norm_type == 'sync_bn':
                logger.info("Converting BatchNorm to SyncBatchNorm")
                self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        # Wrap model with DDP if distributed
        if dist.is_initialized() and dist.get_world_size() > 1:
            find_unused = cfg.get('find_unused_parameters', False)
            self.model = DDP(
                self.model,
                device_ids=[dist.get_rank()],
                find_unused_parameters=find_unused
            )
            logger.info(f"Model wrapped with DDP, world_size={dist.get_world_size()}")

        # Setup EMA (Exponential Moving Average)
        self.use_ema = cfg.get('use_ema', False)
        self.ema = None
        if self.use_ema:
            ema_decay = cfg.get('ema_decay', 0.9998)
            ema_decay_type = cfg.get('ema_decay_type', 'threshold')
            cycle_epoch = cfg.get('cycle_epoch', -1)
            ema_black_list = cfg.get('ema_black_list', None)
            ema_filter_no_grad = cfg.get('ema_filter_no_grad', False)

            # Get the underlying model (not DDP wrapper)
            base_model = self.model.module if isinstance(self.model, DDP) else self.model

            self.ema = ModelEMA(
                base_model,
                decay=ema_decay,
                ema_decay_type=ema_decay_type,
                cycle_epoch=cycle_epoch,
                ema_black_list=ema_black_list,
                ema_filter_no_grad=ema_filter_no_grad
            )
            logger.info(f"EMA enabled with decay={ema_decay}, type={ema_decay_type}")

        # Distributed info
        self._nranks = dist.get_world_size() if dist.is_initialized() else 1
        self._local_rank = dist.get_rank() if dist.is_initialized() else 0

        # Training status dict
        self.status = {}

        # Epoch settings
        self.start_epoch = 0
        self.end_epoch = cfg.get('epoch', 72)

        # Initialize callbacks
        self._init_callbacks()

        # Initialize metrics
        self._init_metrics()
        self._reset_metrics()

        logger.info(f"Trainer initialized in '{self.mode}' mode")

    def _build_data(self, cfg):
        """Build dataset and dataloader using create() factory (Paddle pattern)"""
        capital_mode = self.mode.capitalize()
        dataset_name = '{}Dataset'.format(capital_mode)

        # Build dataset
        if cfg.architecture in MOT_ARCH and self.mode in [
                'eval', 'test'
        ] and cfg.metric not in ['COCO', 'VOC']:
            dataset_name = '{}MOTDataset'.format(capital_mode)
            self.dataset = create(self.cfg[dataset_name])
        else:
            self.dataset = create(self.cfg[dataset_name])

        if cfg.architecture == 'DeepSORT' and self.mode == 'train':
            logger.error('DeepSORT has no need of training on mot dataset.')
            sys.exit(1)

        if cfg.architecture == 'FairMOT' and self.mode == 'eval':
            images = self.parse_mot_images(cfg)
            self.dataset.set_images(images)

        if cfg.architecture == 'JDE' and self.mode == 'train':
            self.cfg['JDEEmbeddingHead'][
                'num_identities'] = self.dataset.num_identities_dict[0]
            # JDE only support single class MOT now.

        if cfg.architecture == 'FairMOT' and self.mode == 'train':
            self.cfg['FairMOTEmbeddingHead'][
                'num_identities_dict'] = self.dataset.num_identities_dict
            # FairMOT support single class and multi-class MOT now.

        # Build dataloader
        if self.mode == 'train':
            reader_name = '{}Reader'.format(capital_mode)
            self.loader = create(self.cfg[reader_name])(
                self.dataset, cfg.worker_num)
        
        if self.mode == 'eval':
            if cfg.architecture == 'FairMOT':
                self.loader = create('EvalMOTReader')(self.dataset, 0)
            elif cfg.architecture == "METRO_Body":
                reader_name = '{}Reader'.format(self.mode.capitalize())
                self.loader = create(reader_name)(self.dataset, cfg.worker_num)
            else:
                # PyTorch equivalent of Paddle's BatchSampler
                from torch.utils.data import BatchSampler, SequentialSampler
                batch_size = self.cfg.EvalReader['batch_size']
                self._eval_batch_sampler = BatchSampler(
                    SequentialSampler(self.dataset),
                    batch_size=batch_size,
                    drop_last=False
                )
                reader_name = '{}Reader'.format(self.mode.capitalize())
                # If metric is VOC, need to be set collate_batch=False.
                if cfg.metric == 'VOC':
                    self.cfg[reader_name]['collate_batch'] = False
                self.loader = create(reader_name)(self.dataset, cfg.worker_num,
                                                  self._eval_batch_sampler)

    def _build_model(self, cfg):
        """Build model using create() factory (Paddle pattern)"""
        # build model
        if 'model' not in self.cfg:
            model_config = dict(self.cfg[cfg.architecture])
            model_config['name'] = cfg.architecture
            self.model = create(model_config)
        else:
            self.model = self.cfg.model
            self.is_loaded_weights = True

        if cfg.architecture == 'YOLOX':
            for k, m in self.model.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3  # for amp(fp16)
                    m.momentum = 0.97  # 0.03 in pytorch

        # reset norm param attr for setting them in optimizer
        if 'reset_norm_param_attr' in cfg and cfg['reset_norm_param_attr']:
            self.model = self.reset_norm_param_attr(
                self.model, weight_attr=None, bias_attr=None)

        # normalize params for deploy
        if 'slim' in cfg and cfg['slim_type'] == 'OFA':
            self.model.model.load_meanstd(cfg['TestReader'][
                'sample_transforms'])
        elif 'slim' in cfg and cfg['slim_type'] == 'Distill':
            self.model.student_model.load_meanstd(cfg['TestReader'][
                'sample_transforms'])
        elif 'slim' in cfg and cfg[
                'slim_type'] == 'DistillPrune' and self.mode == 'train':
            self.model.student_model.load_meanstd(cfg['TestReader'][
                'sample_transforms'])
        else:
            self.model.load_meanstd(cfg['TestReader']['sample_transforms'])

        device = cfg.get('device')
        if device is None:
            use_gpu = cfg.get('use_gpu', False) and torch.cuda.is_available()
            device = torch.device('cuda' if use_gpu else 'cpu')
        self.model.to(device)

        # get Params
        print_params = self.cfg.get('print_params', False)
        if print_params:
            params = sum([
                p.numel() for n, p in self.model.named_parameters()
                if all([x not in n for x in ['_mean', '_variance', 'aux_']])
            ])  # exclude BatchNorm running status
            logger.info('Model Params : {} M.'.format((params / 1e6).numpy()[
                0]))

    def _build_optimizer(self, cfg):
        """Build optimizer and LR scheduler using create() factory (Paddle pattern)"""
        if self.mode != 'train':
            return

        steps_per_epoch = len(self.loader)
        if steps_per_epoch < 1:
            logger.warning(
                "Samples in dataset are less than batch_size, "
                "please set smaller batch_size in TrainReader."
            )
        
        # paddle version
        # # Create LR scheduler
        # self.lr = create('LearningRate')(steps_per_epoch)
        # logger.info(f"LearningRate scheduler created for {steps_per_epoch} steps/epoch")

        # # Create optimizer
        # self.optimizer = create('OptimizerBuilder')(self.lr, self.model)
        # logger.info(f"Optimizer created: {type(self.optimizer).__name__}")

        base_lr = cfg.get('base_lr', 0.001)
        # Create optimizer
        optimizer_config = dict(cfg.get('OptimizerBuilder', {}))
        optimizer_config['name'] = 'OptimizerBuilder'
        self.optimizer = create(optimizer_config)(base_lr, self.model)
        logger.info(f"Optimizer created: {type(self.optimizer).__name__}")

        # Create LR scheduler
        learning_rate_config = dict(cfg.get('LearningRate', {}))
        learning_rate_config['name'] = 'LearningRate'
        self.lr = create(learning_rate_config)(steps_per_epoch, self.optimizer)
        logger.info(f"LearningRate scheduler created for {steps_per_epoch} steps/epoch")


    def _init_callbacks(self):
        """Initialize callbacks (Paddle compatible)"""
        if self.mode == 'train':
            self._callbacks = [LogPrinter(self), Checkpointer(self)]
            # TODO: Add more callbacks based on cfg (VDL, Wandb, etc.)
            self._compose_callback = ComposeCallback(self._callbacks)
            logger.info(f"Initialized {len(self._callbacks)} callbacks for training")
        elif self.mode == 'eval':
            self._callbacks = [LogPrinter(self)]
            self._compose_callback = ComposeCallback(self._callbacks)
        else:
            self._callbacks = []
            self._compose_callback = None

    def _init_metrics(self, validate=False):
        """Initialize metrics (Paddle compatible)"""
        if self.mode == 'test' or (self.mode == 'train' and not validate):
            self._metrics = []
            return

        # TODO: Initialize metrics based on cfg.metric
        # For now, placeholder
        self._metrics = []
        logger.info("Metrics initialized (placeholder)")

    def _reset_metrics(self):
        """Reset all metrics"""
        for metric in self._metrics:
            metric.reset()

    def _convert_cfg_to_dict(self, cfg) -> dict:
        """Convert config object to dictionary for YAML saving"""
        # Simple conversion - can be improved based on actual cfg structure
        if hasattr(cfg, '__dict__'):
            config_dict = {}
            for k, v in cfg.__dict__.items():
                if not k.startswith('_') and v != {}:
                    config_dict[k] = v
            return config_dict
        return dict(cfg)

    def train(self):
        """
        Main training loop (Paddle compatible).

        Executes training for the configured number of epochs.
        """
        logger.info(f"Starting training for {self.end_epoch} epochs")
        logger.info(f"Training on {self._nranks} GPU(s)")

        # Training begin callback
        if self._compose_callback:
            self._compose_callback.on_train_begin(self.status)

        for epoch_id in range(self.start_epoch, self.end_epoch):
            # Set epoch for distributed sampler
            if hasattr(self.loader, 'sampler') and hasattr(self.loader.sampler, 'set_epoch'):
                self.loader.sampler.set_epoch(epoch_id)

            # Update status
            self.status['epoch_id'] = epoch_id
            self.status['mode'] = 'train'

            # Epoch begin callback
            if self._compose_callback:
                self._compose_callback.on_epoch_begin(self.status)

            # Train one epoch
            self._train_epoch(epoch_id)

            # Epoch end callback
            if self._compose_callback:
                self._compose_callback.on_epoch_end(self.status)

        # Training end callback
        if self._compose_callback:
            self._compose_callback.on_train_end(self.status)

        logger.info("Training completed")

    def _train_epoch(self, epoch_id: int):
        """Train for one epoch"""
        self.model.train()

        steps_per_epoch = len(self.loader)
        batch_time_meter = AverageMeter()
        data_time_meter = AverageMeter()

        end = time.time()

        for step_id, batch in enumerate(self.loader):
            # Measure data loading time
            data_time = time.time() - end
            data_time_meter.update(data_time)

            # Update status for callbacks
            self.status['step_id'] = step_id
            self.status['steps_per_epoch'] = steps_per_epoch
            self.status['data_time'] = data_time

            # Step begin callback
            if self._compose_callback:
                self._compose_callback.on_step_begin(self.status)

            # Move data to GPU
            batch = self._prepare_batch(batch)
            if isinstance(batch, dict):
                batch['epoch_id'] = epoch_id

            # Forward pass with AMP
            model_device = next(self.model.parameters()).device
            with torch.amp.autocast(
                    device_type=model_device.type, enabled=self.use_amp):
                outputs = self.model(batch)
                # Model should return dict with 'loss' key
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs
            if not torch.isfinite(loss).all():
                raise FloatingPointError(
                    "Non-finite loss at epoch {}, step {}".format(
                        epoch_id, step_id))

            # Backward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                gradient_norm = self._clip_gradients()

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                gradient_norm = self._clip_gradients()

                self.optimizer.step()

            if not torch.isfinite(torch.as_tensor(gradient_norm)).all():
                raise FloatingPointError(
                    "Non-finite gradient norm at epoch {}, step {}".format(
                        epoch_id, step_id))

            # Update learning rate
            if hasattr(self.lr, 'step'):
                self.lr.step()

            # Update EMA
            if self.use_ema and self.ema:
                base_model = self.model.module if isinstance(self.model, DDP) else self.model
                self.ema.update(base_model)

            # Measure elapsed time
            batch_time = time.time() - end
            batch_time_meter.update(batch_time)
            end = time.time()

            # Update status for callbacks
            self.status['loss'] = loss.item()
            self.status['gradient_norm'] = float(
                torch.as_tensor(gradient_norm).item())
            self.status['learning_rate'] = self.optimizer.param_groups[0]['lr']
            self.status['batch_time'] = batch_time
            self.status['batch_size'] = self._get_batch_size(batch)
            self.status['training_staus'] = {'loss': loss.item()}  # Placeholder

            # Step end callback
            if self._compose_callback:
                self._compose_callback.on_step_end(self.status)

    def _clip_gradients(self):
        """Apply the clipping policy produced by ``OptimizerBuilder``."""
        parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not parameters:
            return torch.tensor(0.0)

        clip_config = getattr(self.optimizer, '_grad_clip', None)
        if clip_config is None and self.cfg.get('grad_clip', 0) > 0:
            clip_config = ('norm', self.cfg.get('grad_clip'))

        if clip_config is None:
            return torch.linalg.vector_norm(torch.stack([
                parameter.grad.detach().norm(2) for parameter in parameters
            ]), 2)

        clip_type, clip_value = clip_config
        if clip_type == 'norm':
            return torch.nn.utils.clip_grad_norm_(parameters, clip_value)
        if clip_type == 'value':
            total_norm = torch.linalg.vector_norm(torch.stack([
                parameter.grad.detach().norm(2) for parameter in parameters
            ]), 2)
            torch.nn.utils.clip_grad_value_(parameters, clip_value)
            return total_norm
        raise ValueError("Unsupported gradient clipping type: {}".format(
            clip_type))

    def _prepare_batch(self, batch):
        """Convert NumPy fields to tensors and move them to the model device."""
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device(self.cfg.get('device', 'cpu'))

        class_index_fields = {'gt_class', 'origin_gt_class'}

        def prepare(value, field_name=None):
            if isinstance(value, torch.Tensor):
                tensor = value.to(device, non_blocking=device.type == 'cuda')
                return tensor.long() if field_name in class_index_fields else tensor
            if isinstance(value, (np.ndarray, np.generic)):
                tensor = torch.as_tensor(value).to(
                    device, non_blocking=device.type == 'cuda')
                return tensor.long() if field_name in class_index_fields else tensor
            if isinstance(value, Mapping):
                return {
                    key: prepare(item, field_name=key)
                    for key, item in value.items()
                }
            if isinstance(value, tuple):
                return tuple(prepare(item, field_name=field_name) for item in value)
            if isinstance(value, list):
                return [prepare(item, field_name=field_name) for item in value]
            return value

        return prepare(batch)

    def _get_batch_size(self, batch) -> int:
        """Get batch size from batch data"""
        if isinstance(batch, dict):
            for v in batch.values():
                if isinstance(v, torch.Tensor):
                    return v.size(0)
        elif isinstance(batch, (list, tuple)):
            for x in batch:
                if isinstance(x, torch.Tensor):
                    return x.size(0)
        elif isinstance(batch, torch.Tensor):
            return batch.size(0)
        return 1

    def load_weights(self, weights: str, ARSL_eval=False):
        """
        Load pretrained weights (Paddle compatible API).

        Args:
            weights: Path to pretrained weights file
            ARSL_eval: ARSL evaluation mode (for compatibility)
        """
        if self.is_loaded_weights:
            return
        self.start_epoch = 0
        load_pretrain_weight(self.model, pretrain_weight=weights, ARSL_eval=ARSL_eval)
        logger.debug(f"Load weights {weights} to start training")

    def resume_weights(self, weights: str):
        """
        Resume training from checkpoint (Paddle compatible API).

        Args:
            weights: Path to checkpoint file
        """
        # Support Distill resume weights
        if hasattr(self.model, 'student_model'):
            self.start_epoch = load_weight(self.model.student_model, weights, self.optimizer)
        else:
            self.start_epoch = load_weight(
                self.model, weights, self.optimizer,
                self.ema if self.use_ema else None
            )
        logger.debug(f"Resume weights of epoch {self.start_epoch}")

    def reset_norm_param_attr(self, layer, **kwargs):
        if isinstance(layer, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            src_state_dict = layer.state_dict()
            if isinstance(layer, nn.BatchNorm2d):
                layer = nn.BatchNorm2d(
                    num_features=layer.num_features,
                    momentum=layer.momentum,
                    epsilon=layer.eps,
                    **kwargs)
            elif isinstance(layer, nn.LayerNorm):
                layer = nn.LayerNorm(
                    normalized_shape=layer.normalized_shape,
                    epsilon=layer.eps,
                    **kwargs)
            else:
                layer = nn.GroupNorm(
                    num_groups=layer.num_groups,
                    num_channels=layer.num_channels,
                    epsilon=layer.eps,
                    **kwargs)
            layer.set_state_dict(src_state_dict)
        else:
            for name, sublayer in layer.named_children():
                new_sublayer = self.reset_norm_param_attr(sublayer, **kwargs)
                if new_sublayer is not sublayer:
                    setattr(layer, name, new_sublayer)

        return layer


class AverageMeter:
    """Compute and store the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.global_avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.global_avg = self.avg
