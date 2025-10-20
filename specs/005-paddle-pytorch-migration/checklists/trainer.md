# Checklist: Trainer Class Requirements Quality

**Purpose**: Validate the completeness, clarity, and consistency of requirements for the Trainer class PyTorch migration from PaddlePaddle. This checklist tests whether the requirements are well-written and ready for implementation - NOT whether the implementation works.

**Created**: 2025-10-20
**Focus**: Configuration-driven initialization, Paddle parity, unused logic branch preservation
**Depth**: Standard (complete Paddle equivalence)
**Audience**: Implementation developer and PR reviewer
**Source**: `RT-DETRv3-paddle/ppdet/engine/trainer.py` (lines 66-299)

---

## Requirement Completeness

### Configuration-Driven Initialization

- [ ] CHK001 - Are all Paddle Trainer `__init__` cfg fields mapped to PyTorch requirements? [Completeness, Gap]
  - Paddle uses: `cfg.amp`, `cfg.amp_level`, `cfg.custom_white_list`, `cfg.custom_black_list`, `cfg.master_grad`, `cfg.uniform_output_enabled`, `cfg.log_ranks`, `cfg.save_dir`, `cfg.worker_num`, `cfg.epoch`, `cfg.use_ema`, `cfg.ema_decay`, etc.
  - Are PyTorch equivalent requirements specified for each?

- [ ] CHK002 - Are dataset construction requirements defined for all Paddle modes ('train', 'eval', 'test')? [Completeness, Spec §FR-004]
  - Paddle: Lines 102-110 create dataset via `create('TrainDataset')`, `create('EvalDataset')`, etc.

- [ ] CHK003 - Are DataLoader construction requirements specified with cfg-driven parameters? [Completeness, Spec §FR-004]
  - Paddle: Lines 121-122 `create('TrainReader')(self.dataset, cfg.worker_num)`
  - Are `worker_num`, `batch_size`, `collate_fn` mapping requirements defined?

- [ ] CHK004 - Are model construction requirements defined for both direct instantiation and pre-loaded model scenarios? [Completeness]
  - Paddle: Lines 135-139 handle `create(cfg.architecture)` vs `self.cfg.model`

- [ ] CHK005 - Are optimizer construction requirements specified as cfg-driven with LearningRate dependency? [Completeness, Spec §FR-004]
  - Paddle: Lines 202-203 `self.lr = create('LearningRate')(steps_per_epoch)` then `create('OptimizerBuilder')(self.lr, self.model)`

- [ ] CHK006 - Are AMP (Automatic Mixed Precision) requirements defined with cfg fields `amp`, `amp_level`, `master_grad`? [Completeness, Gap]
  - Paddle: Lines 74-78, 209-222 handle AMP configuration

- [ ] CHK007 - Are EMA (Exponential Moving Average) requirements specified with all cfg parameters? [Completeness, Spec §FR-004]
  - Paddle: Lines 233-246 use `cfg.use_ema`, `cfg.ema_decay`, `cfg.ema_decay_type`, `cfg.cycle_epoch`, `cfg.ema_black_list`, `cfg.ema_filter_no_grad`

- [ ] CHK008 - Are distributed training setup requirements defined (world_size, rank)? [Completeness, Gap]
  - Paddle: Lines 248-249 `dist.get_world_size()`, `dist.get_rank()`

- [ ] CHK009 - Are callback initialization requirements specified for train/eval/test modes? [Completeness]
  - Paddle: Lines 263-287 initialize different callbacks per mode

- [ ] CHK010 - Are metric initialization requirements defined with cfg-driven metric type? [Completeness, Spec §FR-007]
  - Paddle: Lines 289-299 use `cfg.metric` to select COCO/VOC/LVIS/etc.

### Architecture-Specific Requirements

- [ ] CHK011 - Are requirements defined for MOT (Multi-Object Tracking) architecture special handling? [Completeness, Gap]
  - Paddle: Lines 103-107, 112-132 have MOT-specific logic for JDE/FairMOT/DeepSORT
  - Are these preserved as "unused logic branches" per Spec §FR-004?

- [ ] CHK012 - Are slim/pruning configuration requirements specified? [Completeness, Gap]
  - Paddle: Lines 80-81, 153-163, 206-208 handle PTQ, OFA, Distill, DistillPrune
  - Should these be documented even if unused initially?

- [ ] CHK013 - Are model-specific normalization requirements defined (YOLOX BatchNorm tweaks)? [Completeness, Gap]
  - Paddle: Lines 141-145 modify BatchNorm epsilon/momentum for YOLOX

### Edge Cases & Exception Handling

- [ ] CHK014 - Are requirements defined for empty dataset scenario (samples < batch_size)? [Coverage, Edge Case]
  - Paddle: Lines 198-200 warn if `len(loader) < 1`

- [ ] CHK015 - Are requirements specified for missing cfg fields with fallback defaults? [Coverage, Exception Flow]
  - Paddle uses `cfg.get('key', default)` extensively - are default behaviors documented?

- [ ] CHK016 - Are requirements defined for incompatible architecture-mode combinations? [Coverage, Exception Flow]
  - Paddle: Lines 112-114 error if DeepSORT trains on MOT dataset

---

## Requirement Clarity

### Configuration Field Mapping

- [ ] CHK017 - Is the exact mapping from Paddle cfg keys to PyTorch cfg keys documented? [Clarity, Gap]
  - Example: Paddle's `cfg.amp_level` ('O1'/'O2') → PyTorch's `torch.amp` API equivalence?

- [ ] CHK018 - Are cfg field data types and valid value ranges specified? [Clarity, Gap]
  - Example: `cfg.log_ranks` can be string "0,1,2" or int 0 (Paddle lines 82-86)

- [ ] CHK019 - Is the distinction between "create with factory" vs "direct instantiation" clearly defined? [Clarity, Ambiguity]
  - When does PyTorch Trainer use `create(cfg.X)` vs direct class instantiation?

- [ ] CHK020 - Are "capital mode" string transformations documented as requirements? [Clarity]
  - Paddle: Line 102 `capital_mode = self.mode.capitalize()` → 'Train'/'Eval'/'Test'
  - Used in `'{}Dataset'.format(capital_mode)`, `'{}Reader'.format(capital_mode)`

### Initialization Order Dependencies

- [ ] CHK021 - Is the required initialization sequence documented? [Clarity, Gap]
  - Paddle order: dataset → loader → model → optimizer → EMA → callbacks → metrics
  - Are dependencies (e.g., optimizer needs model, EMA needs model+optimizer) explicit?

- [ ] CHK022 - Are requirements clear for when `steps_per_epoch` must be computed before LR scheduler? [Clarity]
  - Paddle: Line 197 `steps_per_epoch = len(self.loader)` before line 202 `create('LearningRate')(steps_per_epoch)`

### Conditional Logic Clarity

- [ ] CHK023 - Are requirements specified for when to use BatchSampler vs default sampler? [Clarity, Ambiguity]
  - Paddle: Lines 175-182 use custom BatchSampler for eval mode (except MOT/METRO_Body)

- [ ] CHK024 - Are SyncBatchNorm conversion requirements clearly defined with device/norm_type conditions? [Clarity]
  - Paddle: Lines 225-231 check device (npu/xpu/mlu) AND norm_type=='sync_bn' AND multi-rank

---

## Requirement Consistency

### Cross-Module Alignment

- [ ] CHK025 - Are Trainer initialization requirements consistent with data-model.md Entity: Trainer definition? [Consistency, Spec data-model.md §9]
  - data-model.md lists simplified fields (model, optimizer, lr_scheduler, loaders, epochs, dirs)
  - Does spec match Paddle's full cfg-driven complexity?

- [ ] CHK026 - Are Trainer requirements consistent with contracts/training_interface.py? [Consistency, Spec contracts/]
  - TrainerInterface.__init__ has 13 parameters vs Paddle's cfg-driven approach
  - Is this intentional divergence documented?

- [ ] CHK027 - Are optimizer/scheduler construction requirements aligned with research.md API mappings? [Consistency, Spec research.md §2-3]
  - research.md documents AdamW + CosineAnnealing, but Paddle uses factory pattern `create('OptimizerBuilder')`

### Internal Consistency

- [ ] CHK028 - Are mode-dependent requirements (train/eval/test) consistently applied across dataset, loader, callbacks? [Consistency]
  - Paddle: Different logic at lines 102-110 (dataset), 120-182 (loader), 263-287 (callbacks), 289-299 (metrics)

- [ ] CHK029 - Are architecture-specific requirements non-conflicting across MOT variants (JDE, FairMOT, DeepSORT, ByteTrack, CenterTrack)? [Consistency]
  - Paddle: Lines 63, 103-107, 112-132 have overlapping MOT logic

---

## Acceptance Criteria Quality

### Measurability

- [ ] CHK030 - Can "cfg field mapping completeness" be objectively verified with a checklist? [Measurability, Gap]
  - Suggested: Table of (Paddle cfg key, PyTorch cfg key, type, default, validation rule)

- [ ] CHK031 - Are requirements for "initialization success" measurable? [Measurability, Gap]
  - How to verify Trainer.__init__ succeeded? (e.g., all components non-None, no exceptions)

- [ ] CHK032 - Can "Paddle parity" be tested with equivalent cfg inputs producing same internal state? [Measurability, Gap]
  - Suggested: Unit test comparing Paddle vs PyTorch Trainer state after __init__

### Traceability

- [ ] CHK033 - Are requirements traceable to specific Paddle code lines? [Traceability, Gap]
  - This checklist references lines, but does spec.md or FR requirements?

- [ ] CHK034 - Is each cfg field requirement linked to a functional requirement ID? [Traceability, Gap]
  - Example: `cfg.use_ema` → FR-XXX "System must support EMA during training"

---

## Scenario Coverage

### Primary Scenarios

- [ ] CHK035 - Are requirements complete for "Train from scratch on COCO" scenario? [Coverage, Spec §User Story 1]
  - cfg includes: TrainDataset, TrainReader, model architecture, optimizer, LR, epochs

- [ ] CHK036 - Are requirements complete for "Resume training from checkpoint" scenario? [Coverage, Gap]
  - Paddle: Not explicitly in __init__, but implied by `self.start_epoch` (line 253)

- [ ] CHK037 - Are requirements complete for "Evaluation-only mode" scenario? [Coverage]
  - Paddle: Lines 168-182 handle eval mode loader, 277-281 callbacks, 289-299 metrics

### Alternate Scenarios

- [ ] CHK038 - Are requirements defined for "Fine-tuning with pretrained weights" scenario? [Coverage, Gap]
  - Paddle: Not visible in __init__ excerpt, but implied by weight loading (line 40 import)

- [ ] CHK039 - Are requirements defined for "Multi-GPU distributed training" scenario? [Coverage, Spec §FR-004]
  - Paddle: Lines 248-249 setup, but distributed optimizer/model wrapping requirements unclear

- [ ] CHK040 - Are requirements specified for "Mixed architecture training" (e.g., Distill student+teacher)? [Coverage, Gap]
  - Paddle: Lines 156-162 handle slim_type=='Distill' with student_model

### Exception/Error Scenarios

- [ ] CHK041 - Are requirements defined for "Invalid cfg schema" error handling? [Coverage, Exception Flow]
  - What if cfg.mode not in ['train','eval','test']? (Paddle asserts line 69-70)

- [ ] CHK042 - Are requirements specified for "DataLoader creation failure" scenarios? [Coverage, Exception Flow]
  - What if dataset is empty or create() fails?

- [ ] CHK043 - Are requirements defined for "Optimizer creation failure" (incompatible model params)? [Coverage, Exception Flow]
  - Paddle: Line 203 assumes create('OptimizerBuilder') succeeds

### Recovery Scenarios

- [ ] CHK044 - Are requirements specified for "Graceful degradation when EMA unavailable"? [Coverage, Recovery]
  - If `cfg.use_ema=True` but EMA creation fails, should training continue without EMA?

- [ ] CHK045 - Are requirements defined for "Fallback to single-GPU if distributed init fails"? [Coverage, Recovery]
  - Paddle: Lines 248-249 assume dist.get_world_size() works

---

## Non-Functional Requirements

### Performance

- [ ] CHK046 - Are performance requirements specified for Trainer initialization time? [NFR, Gap]
  - Should __init__ complete within X seconds for typical configs?

- [ ] CHK047 - Are memory usage requirements defined for Trainer state? [NFR, Gap]
  - Overhead beyond model/optimizer/data should be < Y MB?

### Maintainability

- [ ] CHK048 - Are requirements for "cfg schema validation" defined to catch misconfigurations early? [NFR, Gap]
  - Paddle uses runtime checks; should PyTorch validate cfg schema at __init__?

- [ ] CHK049 - Are requirements specified for "logging/debugging cfg at initialization"? [NFR]
  - Paddle: Lines 186-193 print model params if `cfg.print_params=True`
  - Should PyTorch log all effective cfg values?

### Compatibility

- [ ] CHK050 - Are backward compatibility requirements defined if cfg schema changes? [NFR, Gap]
  - If new cfg fields added, should old configs still work with defaults?

- [ ] CHK051 - Are requirements specified for "cfg file format compatibility" with Paddle YAML? [NFR, Spec §FR-009]
  - Can PyTorch Trainer accept exact Paddle cfg YAML without modification?

---

## Dependencies & Assumptions

### External Dependencies

- [ ] CHK052 - Are requirements documented for ppdet.core.workspace.create() factory function availability? [Dependency, Gap]
  - Paddle heavily uses `create()` - is PyTorch equivalent required before Trainer implementation?

- [ ] CHK053 - Are requirements specified for callback/metric classes (LogPrinter, Checkpointer, COCOMetric, etc.)? [Dependency, Gap]
  - Trainer __init__ assumes these exist (lines 267-275, 294-299)

- [ ] CHK054 - Are requirements defined for torch.distributed API parity with paddle.distributed? [Dependency, Spec research.md §6.3]
  - Lines 248-249 use Paddle dist API - is PyTorch mapping documented?

### Assumptions

- [ ] CHK055 - Is the assumption "cfg always contains required top-level keys" validated? [Assumption, Gap]
  - Example: Paddle assumes `cfg.architecture`, `cfg.metric`, `cfg.epoch` exist

- [ ] CHK056 - Is the assumption "create() factory never fails" documented and acceptable? [Assumption, Risk]
  - Paddle: No error handling around create() calls (lines 109, 136, 202-203)

- [ ] CHK057 - Is the assumption "len(loader) accurately reflects steps_per_epoch" validated? [Assumption]
  - Paddle: Line 197 - could loader length be misleading if drop_last=True?

---

## Ambiguities & Conflicts

### Ambiguous Requirements

- [ ] CHK058 - Is the meaning of "mode" parameter clearly defined in requirements? [Ambiguity, Gap]
  - Paddle: Lines 67-71 mode affects initialization deeply - are all implications documented?

- [ ] CHK059 - Is "complete Paddle parity" quantified? [Ambiguity, Spec §FR-004]
  - Does it mean 100% of cfg fields, or only "commonly used" ones?

- [ ] CHK060 - Are requirements clear for "unused logic branch preservation"? [Ambiguity, Spec §FR-004]
  - Which branches are "unused" vs "rarely used" vs "deprecated"?
  - Example: Is `cfg.slim_type` OFA/Distill/DistillPrune unused? (lines 153-163)

### Conflicting Requirements

- [ ] CHK061 - Do requirements for "simplified TrainerInterface" conflict with "full Paddle cfg parity"? [Conflict, Spec contracts/ vs research.md]
  - TrainerInterface has hardcoded 13 __init__ params, Paddle uses cfg dict with ~30+ fields

- [ ] CHK062 - Do "configuration-driven" requirements conflict with "type-safe Python API" best practices? [Conflict, Design Decision]
  - Paddle's dict-based cfg vs PyTorch's typed parameters trade-off

### Missing Definitions

- [ ] CHK063 - Is "cfg schema" defined with formal structure (required/optional fields, types, defaults)? [Gap]
  - Needed to implement validation at Trainer.__init__

- [ ] CHK064 - Are "factory function requirements" (create() behavior) formally specified? [Gap]
  - Spec §research.md §1 describes registration, but not create() semantics

- [ ] CHK065 - Is "Paddle cfg → PyTorch cfg" transformation logic defined? [Gap]
  - If Paddle YAML needs preprocessing before PyTorch Trainer, where is this specified?

---

## Summary

**Total Items**: 65
**Categories**:
- Requirement Completeness: 16 items
- Requirement Clarity: 8 items
- Requirement Consistency: 5 items
- Acceptance Criteria Quality: 5 items
- Scenario Coverage: 11 items
- Non-Functional Requirements: 6 items
- Dependencies & Assumptions: 5 items
- Ambiguities & Conflicts: 9 items

**High Priority Gaps** (must resolve before implementation):
- CHK001: Complete Paddle→PyTorch cfg field mapping
- CHK017: Explicit cfg key mapping table
- CHK025-027: Resolve consistency conflicts between spec artifacts
- CHK030-032: Define measurable acceptance criteria
- CHK061-062: Resolve design conflicts (interface simplicity vs cfg parity)
- CHK063-065: Define cfg schema and transformation logic

**Recommended Next Steps**:
1. Create comprehensive Paddle cfg → PyTorch cfg mapping table (addresses CHK001, CHK017, CHK030)
2. Reconcile TrainerInterface with cfg-driven design (addresses CHK026, CHK061)
3. Document unused logic branch preservation policy (addresses CHK060)
4. Add cfg schema validation requirements (addresses CHK048, CHK063)
