"""
Weight conversion utility: PaddlePaddle to PyTorch

This script converts RT-DETRv3 model weights from PaddlePaddle format (.pdparams)
to PyTorch format (.pth), handling parameter name mapping and shape adjustments.

Usage:
    python tools/convert_weights.py \
        --paddle_checkpoint path/to/model.pdparams \
        --config configs/rtdetrv3_r50_6x_coco.yml \
        --output converted.pth \
        --save_mapping mapping.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import warnings

import numpy as np
import torch

try:
    import paddle
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    warnings.warn("PaddlePaddle not installed. Weight conversion will not work.")


logger = logging.getLogger(__name__)


class WeightConverter:
    """Convert PaddlePaddle weights to PyTorch format"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.conversion_stats = {
            'total': 0,
            'converted': 0,
            'skipped': 0,
            'shape_mismatches': []
        }
    
    def load_paddle_checkpoint(self, checkpoint_path: str) -> Dict:
        """
        Load PaddlePaddle checkpoint file.
        
        Args:
            checkpoint_path: Path to .pdparams file
            
        Returns:
            Dictionary of parameter name -> paddle.Tensor
        """
        if not PADDLE_AVAILABLE:
            raise RuntimeError("PaddlePaddle is not installed. Cannot load checkpoint.")
        
        logger.info(f"Loading PaddlePaddle checkpoint from {checkpoint_path}")
        paddle_state = paddle.load(checkpoint_path)
        
        if self.verbose:
            logger.info(f"Loaded {len(paddle_state)} parameters from PaddlePaddle checkpoint")
        
        return paddle_state
    
    def generate_name_mapping(
        self,
        paddle_state: Dict,
        torch_state: Dict,
        manual_overrides: Optional[Dict[str, str]] = None
    ) -> Tuple[Dict[str, str], List[str], List[str]]:
        """
        Generate parameter name mapping from PaddlePaddle to PyTorch.
        
        Args:
            paddle_state: PaddlePaddle state dict
            torch_state: PyTorch state dict
            manual_overrides: Optional manual mapping overrides
            
        Returns:
            Tuple of (mapping dict, unmapped paddle keys, unmapped torch keys)
        """
        paddle_keys = set(paddle_state.keys())
        torch_keys = set(torch_state.keys())
        
        mapping = {}
        
        # Apply manual overrides first
        if manual_overrides:
            for paddle_key, torch_key in manual_overrides.items():
                if paddle_key in paddle_keys and torch_key in torch_keys:
                    mapping[paddle_key] = torch_key
                    paddle_keys.discard(paddle_key)
                    torch_keys.discard(torch_key)
        
        # Automatic mapping based on naming patterns
        mapping.update(self._auto_generate_mapping(paddle_keys, torch_keys))
        
        # Find unmapped keys
        mapped_paddle = set(mapping.keys())
        mapped_torch = set(mapping.values())
        unmapped_paddle = sorted(list(paddle_keys - mapped_paddle))
        unmapped_torch = sorted(list(torch_keys - mapped_torch))
        
        if self.verbose:
            logger.info(f"Generated mapping for {len(mapping)} parameters")
            if unmapped_paddle:
                logger.warning(f"Unmapped PaddlePaddle keys: {len(unmapped_paddle)}")
            if unmapped_torch:
                logger.warning(f"Unmapped PyTorch keys: {len(unmapped_torch)}")
        
        return mapping, unmapped_paddle, unmapped_torch
    
    def _auto_generate_mapping(
        self,
        paddle_keys: set,
        torch_keys: set
    ) -> Dict[str, str]:
        """
        Automatically generate parameter name mapping based on common patterns.
        
        Common mappings:
        - PaddlePaddle: conv2d_1.w_0, bn_1._mean, bn_1._variance
        - PyTorch: features.0.weight, features.0.running_mean, features.0.running_var
        """
        mapping = {}
        
        for paddle_key in list(paddle_keys):
            # Try to find matching torch key by applying naming conventions
            torch_key = self._paddle_to_torch_name(paddle_key)
            
            if torch_key in torch_keys:
                mapping[paddle_key] = torch_key
        
        return mapping
    
    def _paddle_to_torch_name(self, paddle_key: str) -> str:
        """
        Convert PaddlePaddle parameter name to PyTorch convention.
        
        Examples:
            - "backbone.conv1.weight" -> "backbone.conv1.weight" (no change)
            - "bn1._mean" -> "bn1.running_mean"
            - "bn1._variance" -> "bn1.running_var"
            - "conv.w_0" -> "conv.weight"
            - "linear.b_0" -> "linear.bias"
        """
        name = paddle_key
        
        # BatchNorm parameter conversions
        if '._mean' in name:
            name = name.replace('._mean', '.running_mean')
        if '._variance' in name:
            name = name.replace('._variance', '.running_var')
        
        # Weight/bias parameter conversions
        if '.w_0' in name:
            name = name.replace('.w_0', '.weight')
        if '.b_0' in name:
            name = name.replace('.b_0', '.bias')
        
        return name
    
    def convert_tensor(
        self,
        paddle_tensor,
        target_shape: Optional[torch.Size] = None,
        transpose_axes: Optional[Tuple[int, ...]] = None
    ) -> torch.Tensor:
        """
        Convert PaddlePaddle tensor to PyTorch tensor.
        
        Args:
            paddle_tensor: PaddlePaddle tensor
            target_shape: Expected PyTorch tensor shape (for validation)
            transpose_axes: Optional transpose axes if shape conversion needed
            
        Returns:
            PyTorch tensor
        """
        # Convert to numpy
        if hasattr(paddle_tensor, 'numpy'):
            numpy_array = paddle_tensor.numpy()
        else:
            numpy_array = np.array(paddle_tensor)
        
        # Apply transpose if needed
        if transpose_axes is not None:
            numpy_array = np.transpose(numpy_array, transpose_axes)
        
        # Convert to torch
        torch_tensor = torch.from_numpy(numpy_array)
        
        # Validate shape
        if target_shape is not None:
            if torch_tensor.shape != target_shape:
                raise ValueError(
                    f"Shape mismatch: converted tensor has shape {torch_tensor.shape}, "
                    f"expected {target_shape}"
                )
        
        return torch_tensor
    
    def convert_state_dict(
        self,
        paddle_state: Dict,
        torch_state: Dict,
        name_mapping: Dict[str, str],
        strict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Convert PaddlePaddle state dict to PyTorch state dict.
        
        Args:
            paddle_state: Source PaddlePaddle state dict
            torch_state: Target PyTorch state dict (for shape validation)
            name_mapping: Parameter name mapping
            strict: If True, raise error on shape mismatch; if False, skip mismatches
            
        Returns:
            Converted PyTorch state dict
        """
        converted_state = {}
        self.conversion_stats['total'] = len(name_mapping)
        
        for paddle_key, torch_key in name_mapping.items():
            try:
                paddle_param = paddle_state[paddle_key]
                target_shape = torch_state[torch_key].shape
                
                # Convert tensor
                torch_param = self.convert_tensor(
                    paddle_param,
                    target_shape=target_shape if strict else None
                )
                
                converted_state[torch_key] = torch_param
                self.conversion_stats['converted'] += 1
                
                if self.verbose and self.conversion_stats['converted'] % 100 == 0:
                    logger.info(f"Converted {self.conversion_stats['converted']}/{self.conversion_stats['total']} parameters")
            
            except ValueError as e:
                self.conversion_stats['shape_mismatches'].append({
                    'paddle_key': paddle_key,
                    'torch_key': torch_key,
                    'error': str(e)
                })
                
                if strict:
                    raise
                else:
                    logger.warning(f"Skipping {paddle_key} -> {torch_key}: {e}")
                    self.conversion_stats['skipped'] += 1
        
        if self.verbose:
            logger.info(f"Conversion complete: {self.conversion_stats['converted']} converted, "
                       f"{self.conversion_stats['skipped']} skipped")
        
        return converted_state
    
    def save_torch_checkpoint(
        self,
        state_dict: Dict[str, torch.Tensor],
        save_path: str,
        metadata: Optional[Dict] = None
    ):
        """
        Save converted weights as PyTorch checkpoint.
        
        Args:
            state_dict: PyTorch state dict
            save_path: Output .pth file path
            metadata: Optional metadata to include in checkpoint
        """
        checkpoint = {'model': state_dict}
        
        if metadata:
            checkpoint['metadata'] = metadata
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint, save_path)
        logger.info(f"Saved converted checkpoint to {save_path}")


def convert_weights(
    paddle_checkpoint: str,
    output_path: str,
    torch_model = None,
    config_path: Optional[str] = None,
    manual_mapping: Optional[Dict[str, str]] = None,
    save_mapping: Optional[str] = None,
    strict: bool = False
):
    """
    Main function to convert PaddlePaddle checkpoint to PyTorch.
    
    Args:
        paddle_checkpoint: Path to .pdparams file
        output_path: Output .pth file path
        torch_model: Optional PyTorch model instance for structure validation
        config_path: Optional config path to build model automatically
        manual_mapping: Optional manual parameter name mapping
        save_mapping: Optional path to save generated mapping as JSON
        strict: If True, fail on any shape mismatch; if False, skip mismatches
    """
    converter = WeightConverter(verbose=True)
    
    # Load PaddlePaddle checkpoint
    paddle_state = converter.load_paddle_checkpoint(paddle_checkpoint)
    
    # Get PyTorch model structure
    if torch_model is None:
        if config_path is None:
            raise ValueError("Either torch_model or config_path must be provided")
        
        # TODO: Load model from config
        # from rtdetrv3_pytorch.utils.config import load_config
        # config = load_config(config_path)
        # torch_model = build_model(config)
        raise NotImplementedError("Automatic model building from config not yet implemented")
    
    torch_state = torch_model.state_dict()
    
    # Generate name mapping
    name_mapping, unmapped_paddle, unmapped_torch = converter.generate_name_mapping(
        paddle_state,
        torch_state,
        manual_overrides=manual_mapping
    )
    
    # Save mapping if requested
    if save_mapping:
        with open(save_mapping, 'w') as f:
            json.dump({
                'mapping': name_mapping,
                'unmapped_paddle': unmapped_paddle,
                'unmapped_torch': unmapped_torch
            }, f, indent=2)
        logger.info(f"Saved name mapping to {save_mapping}")
    
    # Convert state dict
    converted_state = converter.convert_state_dict(
        paddle_state,
        torch_state,
        name_mapping,
        strict=strict
    )
    
    # Save converted checkpoint
    metadata = {
        'source': 'PaddlePaddle',
        'source_checkpoint': paddle_checkpoint,
        'conversion_stats': converter.conversion_stats
    }
    converter.save_torch_checkpoint(converted_state, output_path, metadata)
    
    logger.info("Conversion completed successfully!")
    return converted_state


def main():
    parser = argparse.ArgumentParser(description="Convert PaddlePaddle checkpoint to PyTorch")
    parser.add_argument('--paddle_checkpoint', type=str, required=True,
                       help='Path to PaddlePaddle .pdparams file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output PyTorch .pth file path')
    parser.add_argument('--config', type=str, default=None,
                       help='Config file to build PyTorch model')
    parser.add_argument('--manual_mapping', type=str, default=None,
                       help='JSON file with manual parameter name mapping')
    parser.add_argument('--save_mapping', type=str, default=None,
                       help='Save generated name mapping to JSON file')
    parser.add_argument('--strict', action='store_true',
                       help='Fail on shape mismatch (default: skip mismatches)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%m/%d %H:%M:%S'
    )
    
    # Load manual mapping if provided
    manual_mapping = None
    if args.manual_mapping:
        with open(args.manual_mapping, 'r') as f:
            manual_mapping = json.load(f)
    
    # Convert weights
    convert_weights(
        paddle_checkpoint=args.paddle_checkpoint,
        output_path=args.output,
        config_path=args.config,
        manual_mapping=manual_mapping,
        save_mapping=args.save_mapping,
        strict=args.strict
    )


if __name__ == '__main__':
    main()
