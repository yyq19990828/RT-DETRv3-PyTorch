# Research: Paddle to PyTorch Weight Conversion

**Feature**: 003-paddle-pytorch-conversion
**Date**: 2025-10-16
**Research Phase**: Phase 0 - Technical Investigation

## Overview

This document consolidates research findings for building a robust weight conversion tool that transforms RT-DETRv3 model weights from PaddlePaddle format (.pdparams) to PyTorch format (.pth). The research addresses parameter name mapping strategies, tensor conversion techniques, numerical validation methods, and memory-efficient processing for large model files.

## Key Research Areas

### 1. Parameter Name Mapping at Scale

**Decision**: Hybrid approach using rule-based auto-mapping with manual override support

**Rationale**:
- PaddlePaddle and PyTorch follow systematic but distinct naming conventions for standard layers
- Rule-based mapping handles 95%+ of common cases automatically (BatchNorm, Conv, Linear)
- Manual overrides provide escape hatch for custom modules or edge cases
- Dictionary-based lookups enable efficient O(1) name translation

**Technical Approach**:
```python
# Common naming pattern transformations
PaddlePaddle → PyTorch:
- "._mean"      → ".running_mean"     (BatchNorm statistics)
- "._variance"  → ".running_var"      (BatchNorm statistics)
- ".w_0"        → ".weight"           (Layer weights)
- ".b_0"        → ".bias"             (Layer biases)
- "._scale"     → ".weight"           (BatchNorm scale)
- "._offset"    → ".bias"             (BatchNorm offset)
```

**Implementation Strategy**:
1. Load both source (Paddle) and target (PyTorch) state_dicts
2. Apply manual overrides first (highest priority)
3. Apply rule-based transformations for unmatched keys
4. Use fuzzy matching (Levenshtein distance) for remaining candidates
5. Report all unmapped keys for manual review

**Alternatives Considered**:
- **Full manual mapping**: Rejected due to maintenance burden for 300+ parameters per model
- **ML-based name inference**: Rejected as overkill; rule-based suffices for systematic naming
- **Hardcoded per-model mappings**: Rejected; violates DRY and requires updates per architecture

**Best Practices from Research**:
- Model introspection via `model.named_parameters()` ensures consistent traversal
- Batch application of mapping rules (vectorized string operations) for performance
- JSON schema for manual mappings enables version control and collaboration

### 2. Tensor Format Conversion

**Decision**: NumPy as intermediate format with explicit shape validation

**Rationale**:
- Both PaddlePaddle and PyTorch tensors can convert to/from NumPy arrays
- NumPy provides framework-agnostic representation for debugging
- Explicit shape validation catches axis ordering issues early
- Supports lazy loading for memory efficiency (via memory-mapped arrays)

**Common Pitfalls Identified**:

| Pitfall | Risk | Mitigation |
|---------|------|------------|
| **Axis transposition** | Conv2D weights use different ordering (NCHW vs NHWC) | Validate shapes against target model; apply transpose only when needed |
| **Non-contiguous memory** | Performance degradation or silent failures | Call `.numpy()` (Paddle) and `torch.from_numpy()` (PyTorch) which handle contiguity |
| **Dtype mismatches** | FP32 → FP16 conversion loses precision | Preserve source dtype by default; explicit conversion only when requested |
| **Missing parameters** | BatchNorm running stats may be optional in training mode | Initialize missing params to framework defaults (running_mean=0, running_var=1) |

**Conversion Pipeline**:
```python
# Paddle Tensor → NumPy → PyTorch Tensor
paddle_param = paddle_state_dict['layer.weight']  # paddle.Tensor
numpy_array = paddle_param.numpy()                 # np.ndarray (contiguous copy)
torch_param = torch.from_numpy(numpy_array)        # torch.Tensor (shares memory with numpy)
```

**Shape Validation Strategy**:
- Compare converted tensor shape against expected PyTorch model shape
- Strict mode: raise ValueError on any mismatch
- Permissive mode: log warning and skip mismatched parameter
- Record all shape mismatches in conversion report for post-mortem analysis

**Alternatives Considered**:
- **Direct Paddle→PyTorch conversion**: No framework-agnostic interface; tight coupling
- **ONNX as intermediate**: Rejected; overkill for weight-only conversion (no graph needed)
- **Binary serialization formats (HDF5, Protobuf)**: Adds dependency complexity without clear benefit

### 3. Numerical Validation

**Decision**: Multi-level validation strategy (parameter-level, module-level, model-level)

**Rationale**:
- Parameter-level checks catch conversion bugs early
- Module-level checks validate weight interactions (e.g., Conv+BN fusion)
- Model-level checks ensure end-to-end correctness
- Layered approach enables incremental debugging

**Validation Levels**:

**Level 1: Parameter Value Preservation**
```python
# Verify converted tensors are numerically identical
numpy_diff = np.abs(paddle_param.numpy() - torch_param.numpy())
assert numpy_diff.max() < 1e-6, "Value mismatch after conversion"
```

**Level 2: Module Output Equivalence**
```python
# Compare module outputs given same input
test_input = torch.randn(1, 3, 224, 224)
paddle_module.eval()
torch_module.eval()

paddle_output = paddle_module(test_input)
torch_output = torch_module(test_input)

torch.testing.assert_close(
    torch_output, paddle_output,
    atol=1e-5, rtol=1e-4,
    msg="Module outputs differ beyond tolerance"
)
```

**Level 3: Model Inference Validation**
```python
# End-to-end validation on real data
from pycocotools.coco import COCO

# Load same input image
input_tensor = preprocess_image('coco_val_sample.jpg')

# Compare predictions
paddle_predictions = paddle_model(input_tensor)
torch_predictions = torch_model(input_tensor)

# Validate bbox coordinates, scores, classes
assert_predictions_match(paddle_predictions, torch_predictions, tol=1e-5)
```

**Tolerance Guidelines** (from Constitution Principle I):
- **FP32 operations**: 1e-5 absolute tolerance (default)
- **Accumulated operations** (e.g., BatchNorm stats over batches): 1e-4 relative tolerance
- **Stochastic layers** (Dropout, data augmentation): Disable or use fixed seeds

**Best Practices from Research**:
- Use `torch.manual_seed()` and `paddle.seed()` for deterministic testing
- Test multiple input shapes (1x3x224x224, 2x3x384x384, etc.)
- Validate both forward pass (outputs) and backward pass (gradients) where applicable
- Create regression test suite with golden outputs from PaddlePaddle reference

**Alternatives Considered**:
- **Statistical validation only** (mean/std comparison): Rejected; misses subtle errors
- **Checksum/hash validation**: Rejected; requires exact binary match (impossible across frameworks)
- **Visual inspection**: Rejected as primary method; unreliable and not automatable

### 4. Memory-Efficient Strategies for Large Models

**Decision**: Lazy loading with chunked processing and optional mmap support

**Rationale**:
- RT-DETRv3 models range 92MB (r18vd) to 182MB (r50vd)
- Memory-mapped files avoid loading entire checkpoint into RAM
- Chunked processing enables conversion of even larger models (>1GB)
- Supports low-memory environments (e.g., CPU-only machines with 8GB RAM)

**Implementation Strategy**:

**Approach 1: Standard Loading (Models <500MB)**
```python
# Load entire state_dict into memory (simplest, fastest for small models)
paddle_state = paddle.load(checkpoint_path)  # ~2x file size in memory
torch_state = convert_all_params(paddle_state)
torch.save(torch_state, output_path)
```
**Memory Usage**: ~2x source file size (182MB → 364MB peak)

**Approach 2: Chunked Processing (Models >500MB)**
```python
# Process parameters in batches to limit memory usage
paddle_state = paddle.load(checkpoint_path)
torch_state = {}

for param_name in paddle_state.keys():
    # Convert one parameter at a time
    torch_param = convert_param(paddle_state[param_name])
    torch_state[param_name] = torch_param

    # Optional: delete source param to free memory
    del paddle_state[param_name]

torch.save(torch_state, output_path)
```
**Memory Usage**: ~1.2x source file size (constant overhead for largest single tensor)

**Approach 3: Memory-Mapped Loading (Models >1GB)**
```python
# Use numpy.load with mmap_mode for zero-copy access
paddle_state = paddle.load(checkpoint_path)
numpy_arrays = {k: v.numpy() for k, v in paddle_state.items()}

# Save to .npz with memory mapping
np.savez(temp_npz_path, **numpy_arrays)
mmap_data = np.load(temp_npz_path, mmap_mode='r')

# Convert on-demand without loading entire file
torch_state = {}
for key in mmap_data.keys():
    torch_state[key] = torch.from_numpy(mmap_data[key].copy())
```
**Memory Usage**: ~200MB constant (only active tensor in memory)

**Additional Memory Optimizations**:
- **Dtype downcast**: Convert FP32 → FP16 post-conversion (halves storage, requires validation)
- **Compression**: Use `torch.save(..., _use_new_zipfile_serialization=True)` for 10-20% size reduction
- **Garbage collection**: Explicit `del` statements + `gc.collect()` after processing large tensors
- **Generator-based conversion**: Yield converted params one at a time for streaming writes

**Performance Metrics** (from Specification SC-001, SC-006):
- Conversion time target: <2 minutes for 182MB model (r50vd)
- Memory overhead target: ≤2x source file size (364MB peak for r50vd)

**Best Practices from Research**:
- Profile memory usage with `tracemalloc` or `memory_profiler` during development
- Provide `--memory-efficient` CLI flag for users with constrained environments
- Document memory requirements clearly in tool help text

**Alternatives Considered**:
- **Stream-based conversion**: Rejected; PyTorch state_dict format requires full dict at save time
- **Incremental save**: Rejected; would require custom checkpoint format (breaks compatibility)
- **External memory (swap)**: Rejected; performance unacceptable (10-100x slower)

## Technology Choices

### Core Dependencies

| Dependency | Version | Rationale |
|------------|---------|-----------|
| **PyTorch** | ≥2.0.0 | Target framework; 2.0+ for `torch.compile` compatibility |
| **PaddlePaddle** | ≥2.4.0 | Source framework; required to load .pdparams files |
| **NumPy** | ≥1.21.0 | Intermediate tensor representation; universal compatibility |
| **pytest** | ≥7.0.0 | Testing framework per constitution; `pytest-xdist` for parallel tests |

### Optional Dependencies

| Dependency | Version | Use Case |
|------------|---------|----------|
| **tqdm** | ≥4.60.0 | Progress bars for batch conversion |
| **PyYAML** | ≥6.0 | Config file parsing (if model auto-build needed) |
| **h5py** | ≥3.7.0 | HDF5 support for alternative checkpoint formats |

### Python Version Support

- **Minimum**: Python 3.8 (per constitution: compatibility requirement)
- **Recommended**: Python 3.11 (per constitution: performance benefits)
- **Tested**: Python 3.8, 3.10, 3.11 (CI matrix)

## Integration Patterns

### CLI Interface Design

**Command Structure**:
```bash
# Basic conversion
python tools/convert_weights.py \
    --input pretrained_models/paddle/rtdetrv3_r50vd_6x_coco.pdparams \
    --output pretrained_models/pytorch/rtdetrv3_r50vd_6x_coco.pth \
    --model-config configs/rtdetrv3_r50vd.yml

# With mapping export and validation
python tools/convert_weights.py \
    --input pretrained_models/paddle/rtdetrv3_r50vd_6x_coco.pdparams \
    --output pretrained_models/pytorch/rtdetrv3_r50vd_6x_coco.pth \
    --save-mapping mappings/r50vd_mapping.json \
    --strict  # Fail on any shape mismatch

# Batch conversion
python tools/convert_weights.py \
    --batch pretrained_models/paddle/*.pdparams \
    --output-dir pretrained_models/pytorch/
```

**Arguments**:
- `--input, -i`: Source .pdparams file (required)
- `--output, -o`: Target .pth file (required)
- `--model-config, -c`: YAML config to build PyTorch model (required for validation)
- `--manual-mapping, -m`: JSON file with custom parameter name mappings (optional)
- `--save-mapping, -s`: Export generated mapping to JSON (optional)
- `--strict`: Fail on shape mismatches (default: false, skip mismatches)
- `--batch, -b`: Glob pattern for batch conversion (optional)
- `--output-dir, -d`: Output directory for batch mode (optional)
- `--memory-efficient`: Use chunked processing for large models (optional)
- `--validate`: Run numerical validation after conversion (optional)
- `--tolerance`: Numerical tolerance for validation (default: 1e-5)

### Programmatic API

```python
from tools.weight_conversion import WeightConverter

# Initialize converter
converter = WeightConverter(verbose=True)

# Load source checkpoint
paddle_state = converter.load_paddle_checkpoint('model.pdparams')

# Generate name mapping (with auto-detection)
mapping, unmapped_paddle, unmapped_torch = converter.generate_name_mapping(
    paddle_state,
    torch_model.state_dict(),
    manual_overrides={'custom.layer': 'custom_layer'}
)

# Convert state dict
torch_state = converter.convert_state_dict(
    paddle_state,
    torch_model.state_dict(),
    mapping,
    strict=False
)

# Save converted checkpoint
converter.save_torch_checkpoint(
    torch_state,
    'converted.pth',
    metadata={'source': 'rtdetrv3_paddle'}
)

# Access conversion statistics
print(converter.conversion_stats)
# {'total': 315, 'converted': 312, 'skipped': 3, 'shape_mismatches': [...]}
```

## Known Limitations and Edge Cases

### Limitation 1: Architecture Dependence
**Issue**: Conversion requires target PyTorch model instance for shape validation
**Impact**: Cannot convert weights without corresponding PyTorch model definition
**Workaround**: Provide `--no-validate` mode for shape-agnostic conversion (skips validation)

### Limitation 2: Custom Operators
**Issue**: PaddlePaddle custom ops (C++/CUDA extensions) have no PyTorch equivalent
**Impact**: Models using custom ops cannot be fully converted
**Mitigation**: Document all RT-DETRv3 custom ops; provide PyTorch reimplementations

### Limitation 3: Training State
**Issue**: Optimizer states, learning rate schedulers not converted
**Impact**: Converted checkpoints suitable for inference/fine-tuning only (not resume training)
**Workaround**: Out of scope per specification; document limitation clearly

### Edge Case 1: Mixed Precision Models
**Issue**: Models trained with AMP (FP16/BF16) may have FP32 master weights
**Mitigation**: Detect dtype from checkpoint metadata; preserve original precision

### Edge Case 2: Distributed Training Artifacts
**Issue**: Multi-GPU training may store wrapped models (e.g., `model.module.layer`)
**Mitigation**: Strip wrapper prefixes during name mapping (configurable)

### Edge Case 3: Pruned/Quantized Models
**Issue**: Pruning/quantization metadata not transferable between frameworks
**Mitigation**: Convert weights only; document that post-training modifications lost

## Validation Strategy

### Unit Tests
- `test_name_mapping.py`: Test all naming convention rules
- `test_tensor_conversion.py`: Test numpy conversion pipeline
- `test_shape_validation.py`: Test strict/permissive shape checking
- `test_cli.py`: Test command-line interface argument parsing

### Integration Tests
- `test_full_conversion_r18vd.py`: Convert r18vd model, validate numerically
- `test_full_conversion_r34vd.py`: Convert r34vd model, validate numerically
- `test_full_conversion_r50vd.py`: Convert r50vd model, validate numerically
- `test_batch_conversion.py`: Test batch mode with multiple files

### Numerical Validation Tests
- `test_parameter_preservation.py`: Verify converted weights match source values
- `test_module_equivalence.py`: Compare backbone outputs (Paddle vs PyTorch)
- `test_inference_equivalence.py`: Compare full model predictions on COCO samples

### Performance Tests
- `test_conversion_speed.py`: Verify <2 min conversion time for r50vd
- `test_memory_usage.py`: Verify ≤2x memory overhead during conversion

## Implementation Roadmap

### Phase 1: Core Conversion Engine (Priority: P1)
- Enhance `WeightConverter` class with robust name mapping
- Implement tensor conversion with shape validation
- Add conversion statistics tracking
- **Validation**: Unit tests pass; manual smoke test with r50vd

### Phase 2: CLI Interface (Priority: P1)
- Implement argparse-based CLI with all specified options
- Add progress logging and error reporting
- Support batch conversion mode
- **Validation**: CLI tests pass; convert all 3 model variants successfully

### Phase 3: Numerical Validation (Priority: P2)
- Implement parameter-level validation
- Implement module-level validation (requires PyTorch model)
- Implement model-level validation (requires test dataset)
- **Validation**: All numerical tests pass with 1e-5 tolerance

### Phase 4: Memory Optimization (Priority: P2)
- Implement chunked processing mode
- Add memory profiling and optimization
- **Validation**: Convert r50vd with <400MB memory usage

### Phase 5: Documentation (Priority: P3)
- Write user guide with examples
- Document all CLI options
- Create troubleshooting guide for common issues
- **Validation**: Documentation review by stakeholders

## References

### Research Sources
1. **Perplexity Research**: "Best practices for weight conversion tools" - Parameter mapping strategies, tensor conversion pitfalls, validation approaches, memory optimization
2. **DeepWiki (PaddlePaddle)**: Parameter naming conventions in PaddlePaddle framework
3. **DeepWiki (PyTorch)**: State dict serialization and checkpoint handling best practices
4. **Existing Draft**: `tools/convert_weights.py` - Reference implementation with WeightConverter class

### Related Documentation
- RT-DETRv3 Constitution: `.specify/memory/constitution.md` (Validation requirements)
- Feature Specification: `specs/003-paddle-pytorch-conversion/spec.md`
- Tech Report: `tech-report.md` (PaddlePaddle codebase analysis)

### External Resources
- PyTorch State Dict Guide: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_models_for_inference.html
- PaddlePaddle Model Save/Load: https://www.paddlepaddle.org.cn/documentation/docs/en/guides/model_convert/save_load_en.html

---

**Research Complete**: All unknowns resolved. Ready to proceed to Phase 1 (Design & Contracts).
