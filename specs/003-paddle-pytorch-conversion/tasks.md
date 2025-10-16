# Tasks: Paddle to PyTorch Weight Conversion

**Input**: Design documents from `/specs/003-paddle-pytorch-conversion/`
**Prerequisites**: plan.md, spec.md, data-model.md, contracts/cli-interface.md, research.md

**Tests**: Not explicitly requested in specification - following constitution validation-driven approach with pytest framework

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions
- **Single project structure** (utility tool): `tools/`, `tests/` at repository root
- Paths based on plan.md structure

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and module structure

- [X] T001 Create tools/weight_conversion/ module directory with __init__.py
- [X] T002 [P] Create tests/test_weight_conversion/ directory with __init__.py
- [X] T003 [P] Create tests/test_weight_conversion/fixtures/ directory for test data
- [X] T004 [P] Create tests/integration/ directory with __init__.py
- [X] T005 [P] Configure pytest with pytest.ini or pyproject.toml (test discovery, markers)
- [X] T006 [P] Add Python dependencies to requirements: torch>=2.0, paddlepaddle>=2.4, numpy>=1.21, pytest>=7.0
- [X] T007 [P] Create pretrained_models/pytorch/ output directory

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T008 Implement data model classes in tools/weight_conversion/models.py (CheckpointFile, Parameter, ParameterMapping, ConversionSession, ConversionConfig, ConversionStatistics, ShapeMismatch, DtypeConversion)
- [X] T009 [P] Implement tensor conversion utilities in tools/weight_conversion/tensor_utils.py (paddle_to_numpy, numpy_to_torch, validate_tensor_shape, detect_dtype)
- [X] T010 [P] Implement logging infrastructure in tools/weight_conversion/__init__.py (configure_logging, get_logger)
- [X] T011 [P] Create test fixtures in tests/test_weight_conversion/fixtures/ (sample_paddle.pdparams, expected_mappings.json)

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Basic Weight Conversion (Priority: P1) 🎯 MVP

**Goal**: Enable users to convert .pdparams files to .pth format with automatic parameter name mapping and tensor conversion

**Independent Test**: Provide rtdetrv3_r50vd_6x_coco.pdparams, run conversion tool, verify output .pth file contains correctly converted weights loadable by PyTorch

### Tests for User Story 1

**NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T012 [P] [US1] Unit test for paddle checkpoint loading in tests/test_weight_conversion/test_converter.py::test_load_paddle_checkpoint
- [X] T013 [P] [US1] Unit test for tensor conversion (paddle→numpy→torch) in tests/test_weight_conversion/test_tensor_utils.py::test_convert_tensor
- [X] T014 [P] [US1] Unit test for parameter name mapping generation in tests/test_weight_conversion/test_name_mapping.py::test_generate_name_mapping
- [X] T015 [P] [US1] Unit test for torch checkpoint saving in tests/test_weight_conversion/test_converter.py::test_save_torch_checkpoint
- [X] T016 [P] [US1] Integration test for end-to-end conversion in tests/integration/test_full_conversion.py::test_convert_r50vd_model

### Implementation for User Story 1

- [X] T017 [P] [US1] Implement CheckpointFile loading logic in tools/weight_conversion/converter.py::load_paddle_checkpoint (uses paddle.load, validates format)
- [X] T018 [P] [US1] Implement automatic parameter name mapping in tools/weight_conversion/name_mapping.py (NameMapper class with _paddle_to_torch_name, apply_naming_rules: ._mean→.running_mean, ._variance→.running_var, .w_0→.weight, .b_0→.bias)
- [X] T019 [US1] Implement WeightConverter.convert_tensor in tools/weight_conversion/converter.py (paddle tensor → numpy → torch tensor with shape validation)
- [X] T020 [US1] Implement WeightConverter.convert_state_dict in tools/weight_conversion/converter.py (iterates mappings, converts tensors, tracks statistics)
- [X] T021 [US1] Implement CheckpointFile saving logic in tools/weight_conversion/converter.py::save_torch_checkpoint (saves state_dict + metadata)
- [X] T022 [US1] Implement ConversionSession orchestration in tools/weight_conversion/converter.py::WeightConverter.convert (coordinates load→map→convert→save workflow)
- [X] T023 [US1] Add conversion statistics tracking in WeightConverter (total, converted, skipped counts)
- [X] T024 [US1] Add progress logging for conversion (log every 100 parameters converted)
- [X] T025 [US1] Implement basic error handling (file not found, invalid checkpoint format, shape mismatch in strict mode)

**Checkpoint**: At this point, User Story 1 should be fully functional - users can convert .pdparams to .pth

---

## Phase 4: User Story 2 - Parameter Name Mapping Validation (Priority: P2)

**Goal**: Enable users to export and inspect parameter name mappings, identify unmapped parameters for debugging

**Independent Test**: Run conversion with --save-mapping flag, verify exported JSON accurately reflects transformations and lists unmapped parameters

### Tests for User Story 2

- [X] T026 [P] [US2] Unit test for manual mapping override in tests/test_weight_conversion/test_name_mapping.py::test_apply_manual_mappings
- [X] T027 [P] [US2] Unit test for mapping export to JSON in tests/test_weight_conversion/test_name_mapping.py::test_export_mapping_to_json
- [X] T028 [P] [US2] Unit test for unmapped parameter detection in tests/test_weight_conversion/test_name_mapping.py::test_identify_unmapped_parameters
- [X] T029 [P] [US2] Integration test for mapping export workflow in tests/integration/test_full_conversion.py::test_conversion_with_mapping_export

### Implementation for User Story 2

- [X] T030 [P] [US2] Implement manual mapping loader in tools/weight_conversion/name_mapping.py::NameMapper.load_manual_mappings (reads JSON file, validates schema)
- [X] T031 [P] [US2] Implement manual mapping application in tools/weight_conversion/name_mapping.py::NameMapper.apply_manual_overrides (applies before auto-mapping)
- [X] T032 [US2] Implement mapping export functionality in tools/weight_conversion/name_mapping.py::NameMapper.export_to_json (generates exported mapping JSON with session metadata, mapping types, confidence scores)
- [X] T033 [US2] Implement unmapped parameter identification in tools/weight_conversion/name_mapping.py::NameMapper.find_unmapped_keys (compares source keys vs mapped keys, target keys vs populated keys)
- [X] T034 [US2] Add mapping export to ConversionSession in tools/weight_conversion/converter.py::WeightConverter.convert (optional --save-mapping argument)
- [X] T035 [US2] Add unmapped parameter reporting to ConversionStatistics in tools/weight_conversion/models.py (unmapped_source_keys, unmapped_target_keys lists)
- [X] T036 [US2] Update logging to warn about unmapped parameters during conversion

**Checkpoint**: At this point, User Stories 1 AND 2 work independently - users can convert AND inspect mappings

---

## Phase 5: User Story 4 - Shape Mismatch Handling (Priority: P2)

**Goal**: Provide clear error messages for shape mismatches, support strict/permissive modes, generate detailed mismatch reports

**Independent Test**: Provide mismatched model architectures, run in strict mode (should fail with clear error), run in permissive mode (should skip with warning and list in report)

### Tests for User Story 4

- [X] T037 [P] [US4] Unit test for shape validation in tests/test_weight_conversion/test_tensor_utils.py::test_validate_tensor_shape_* (match, mismatch_non_strict, mismatch_strict, check_shape_compatibility_*)
- [X] T038 [P] [US4] Unit test for strict mode error handling in tests/test_weight_conversion/test_converter.py::test_convert_tensor_shape_mismatch_strict
- [X] T039 [P] [US4] Unit test for permissive mode skip behavior in tests/test_weight_conversion/test_converter.py::test_convert_tensor_shape_mismatch_permissive
- [X] T040 [P] [US4] Integration test for shape mismatch reporting in tests/integration/test_full_conversion.py::test_conversion_with_shape_mismatches

### Implementation for User Story 4

- [X] T041 [P] [US4] Implement shape validation logic in tools/weight_conversion/validation.py::ShapeValidator (compare_shapes, detect_mismatch_severity: ERROR vs WARNING) - integrated in tensor_utils.py
- [X] T042 [P] [US4] Implement ShapeMismatch recording in tools/weight_conversion/models.py (parameter_name, source_shape, target_shape, severity, suggested_fix)
- [X] T043 [US4] Add strict mode handling to WeightConverter.convert_tensor in tools/weight_conversion/converter.py (raise ValueError on shape mismatch if strict=True)
- [X] T044 [US4] Add permissive mode handling to WeightConverter.convert_tensor in tools/weight_conversion/converter.py (log warning, skip parameter, continue if strict=False)
- [X] T045 [US4] Update ConversionStatistics to track shape_mismatches list in tools/weight_conversion/models.py
- [X] T046 [US4] Add shape mismatch reporting to final conversion summary (list all mismatched parameters with shapes)
- [ ] T047 [US4] Implement suggested fix generation for shape mismatches in tools/weight_conversion/validation.py::ShapeValidator.suggest_fix (detect transpose, reshape, padding needs) - deferred to Phase 8

**Checkpoint**: At this point, User Stories 1, 2, AND 4 work independently - robust error handling for shape issues

---

## Phase 6: User Story 3 - Batch Conversion for Multiple Models (Priority: P3)

**Goal**: Enable users to convert multiple .pdparams files in one command using glob patterns or directory input

**Independent Test**: Provide directory with 3 .pdparams files, run batch conversion, verify all 3 .pth files generated correctly

### Tests for User Story 3

- [ ] T048 [P] [US3] Unit test for glob pattern parsing in tests/test_weight_conversion/test_cli.py::test_parse_batch_pattern
- [ ] T049 [P] [US3] Unit test for batch file discovery in tests/test_weight_conversion/test_cli.py::test_discover_batch_files
- [ ] T050 [P] [US3] Integration test for batch conversion workflow in tests/integration/test_full_conversion.py::test_batch_convert_multiple_models
- [ ] T051 [P] [US3] Integration test for batch conversion with one failure in tests/integration/test_full_conversion.py::test_batch_conversion_continues_on_failure

### Implementation for User Story 3

- [ ] T052 [P] [US3] Implement batch file discovery in tools/weight_conversion/converter.py::WeightConverter.discover_batch_files (accepts glob pattern, returns list of .pdparams paths)
- [ ] T053 [US3] Implement batch conversion orchestration in tools/weight_conversion/converter.py::WeightConverter.batch_convert (iterates files, converts each, aggregates statistics)
- [ ] T054 [US3] Add batch conversion error handling in tools/weight_conversion/converter.py (continue on individual file failure, track successes/failures)
- [ ] T055 [US3] Add batch conversion progress reporting (log "Converting file X of Y...", aggregate conversion times)
- [ ] T056 [US3] Implement batch output path generation in tools/weight_conversion/converter.py (auto-generate output paths based on input filenames)
- [ ] T057 [US3] Add batch conversion summary statistics (total files, successful, failed, total time)

**Checkpoint**: All user stories (1, 2, 3, 4) now work independently - full feature set complete

---

## Phase 7: CLI Interface (Cross-Cutting)

**Purpose**: Provide command-line interface for all user stories per contracts/cli-interface.md

- [X] T058 [P] Implement CLI argument parser in tools/weight_conversion/cli.py::create_argument_parser (--input, --output, --model-config, --manual-mapping, --save-mapping, --strict, --permissive, --no-validate, --validate-numerical, --tolerance, --batch, --output-dir, --memory-efficient, --force, --log-level, --quiet, --version, --help per CLI contract)
- [X] T059 [P] Implement CLI argument validation in tools/weight_conversion/cli.py::validate_arguments (check mutual exclusivity: --strict vs --permissive, --input vs --batch; check required dependencies: --batch requires --output-dir, --validate-numerical requires --model-config)
- [X] T060 [P] Implement CLI main entry point in tools/weight_conversion/cli.py::main (parse args, configure logging, create ConversionConfig, call WeightConverter)
- [X] T061 [P] Update tools/convert_weights.py to use new CLI interface (replace existing main() with import from cli.py)
- [ ] T062 [P] Implement progress bar for interactive mode in tools/weight_conversion/cli.py (use tqdm if available, show "Converting parameters: X/Y") - deferred to Phase 8
- [X] T063 [P] Implement log-based progress for non-interactive mode in tools/weight_conversion/cli.py (log every 100 params when stdout not a TTY)
- [X] T064 [P] Implement exit code handling in tools/weight_conversion/cli.py (0=success, 1=error, 2=invalid args, 3=validation failed, 130=interrupted)
- [X] T065 [P] Add environment variable support in tools/weight_conversion/cli.py (PADDLE_CONV_LOG_LEVEL, PADDLE_CONV_MEMORY_LIMIT)

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [X] T066 [P] Add comprehensive docstrings to all public classes and methods (Google style)
- [X] T067 [P] Add type hints to all function signatures (use typing module, PEP 484 compliant)
- [ ] T068 [P] Implement memory-efficient mode in tools/weight_conversion/converter.py::WeightConverter (chunked processing, batch_size=100 params, explicit gc.collect())
- [ ] T069 [P] Add numerical validation support in tools/weight_conversion/validation.py::NumericalValidator (compare converted values vs source, check tolerance)
- [ ] T070 [P] Implement performance profiling in tools/weight_conversion/converter.py (track conversion time, memory usage via tracemalloc)
- [X] T071 [P] Add checkpoint metadata embedding in tools/weight_conversion/converter.py::save_torch_checkpoint (source file, timestamp, tool version, conversion stats, validation results)
- [ ] T072 [P] Create comprehensive README.md for tools/weight_conversion/ module (architecture overview, usage examples, API reference)
- [X] T073 [P] Add CLI usage examples to tools/convert_weights.py docstring (basic conversion, batch mode, strict mode, mapping export)
- [X] T074 [P] Run pytest test suite and ensure all tests pass (pytest tests/ -v) - 40 passed, 1 skipped
- [ ] T075 [P] Validate conversion tool against all 3 model variants (r18vd, r34vd, r50vd) per quickstart.md
- [ ] T076 [P] Verify performance targets (SC-001: <2 min for 182MB; SC-006: ≤2x memory usage)
- [ ] T077 [P] Run code quality checks (ruff or flake8 for linting, black for formatting)
- [ ] T078 Validate quickstart.md instructions (follow guide step-by-step, verify all examples work)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-6)**: All depend on Foundational phase completion
  - User Story 1 (P1) → User Story 2 (P2) has soft dependency (US2 extends US1 with mapping export)
  - User Story 1 (P1) → User Story 4 (P2) has soft dependency (US4 extends US1 with error handling)
  - User Story 3 (P3) depends on User Story 1 (P1) completion (batch mode wraps single conversion)
  - Can proceed in priority order: US1 → US2/US4 (parallel) → US3
- **CLI Interface (Phase 7)**: Depends on all user stories being complete
- **Polish (Phase 8)**: Depends on CLI interface and all user stories

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories (MVP)
- **User Story 2 (P2)**: Can start after User Story 1 - Extends US1 with mapping export functionality
- **User Story 4 (P2)**: Can start after User Story 1 - Extends US1 with shape mismatch handling
- **User Story 3 (P3)**: Depends on User Story 1 completion - Wraps single conversion for batch processing

### Within Each User Story

- Tests MUST be written and FAIL before implementation (validation-driven per constitution)
- Unit tests before integration tests
- Models/utilities before services
- Core conversion logic before orchestration
- Error handling after happy path
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Within User Story 1: Tests (T012-T016) can run in parallel, Implementation (T017, T018) can run in parallel
- Within User Story 2: Tests (T026-T029) can run in parallel, Implementation (T030, T031) can run in parallel
- Within User Story 4: Tests (T037-T040) can run in parallel, Implementation (T041, T042) can run in parallel
- Within User Story 3: Tests (T048-T051) can run in parallel, Implementation (T052) can run in parallel
- User Stories 2 and 4 can be developed in parallel after US1 completes
- All CLI Interface tasks (T058-T065) can run in parallel after user stories complete
- All Polish tasks (T066-T077) can run in parallel

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together:
Task: "Unit test for paddle checkpoint loading in tests/test_weight_conversion/test_converter.py::test_load_paddle_checkpoint"
Task: "Unit test for tensor conversion in tests/test_weight_conversion/test_tensor_utils.py::test_convert_tensor"
Task: "Unit test for parameter name mapping in tests/test_weight_conversion/test_name_mapping.py::test_generate_name_mapping"
Task: "Unit test for torch checkpoint saving in tests/test_weight_conversion/test_converter.py::test_save_torch_checkpoint"
Task: "Integration test for end-to-end conversion in tests/integration/test_full_conversion.py::test_convert_r50vd_model"

# Launch parallel implementation tasks:
Task: "Implement CheckpointFile loading logic in tools/weight_conversion/converter.py::load_paddle_checkpoint"
Task: "Implement automatic parameter name mapping in tools/weight_conversion/name_mapping.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T007)
2. Complete Phase 2: Foundational (T008-T011) - **CRITICAL BLOCKER**
3. Complete Phase 3: User Story 1 (T012-T025)
4. **STOP and VALIDATE**: Test User Story 1 independently
   - Convert rtdetrv3_r50vd_6x_coco.pdparams
   - Load converted .pth into PyTorch model
   - Verify all parameters loaded successfully
5. Deploy/demo if ready - **THIS IS THE MVP**

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → **Deploy/Demo (MVP!)**
3. Add User Story 2 → Test independently → Deploy/Demo (adds mapping transparency)
4. Add User Story 4 → Test independently → Deploy/Demo (adds robust error handling)
5. Add User Story 3 → Test independently → Deploy/Demo (adds batch processing efficiency)
6. Add CLI Interface → Complete feature
7. Polish → Production ready
8. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together (T001-T011)
2. Once Foundational is done:
   - **Developer A**: User Story 1 (T012-T025) - **PRIORITY**
3. After US1 completes:
   - **Developer B**: User Story 2 (T026-T036)
   - **Developer C**: User Story 4 (T037-T047)
4. After US2/US4 complete:
   - **Developer D**: User Story 3 (T048-T057)
5. After all user stories:
   - **All developers**: CLI Interface (T058-T065) in parallel
   - **All developers**: Polish (T066-T078) in parallel

---

## Task Statistics

**Total Tasks**: 78
- **Phase 1 (Setup)**: 7 tasks
- **Phase 2 (Foundational)**: 4 tasks
- **Phase 3 (US1 - Basic Conversion)**: 14 tasks (5 tests + 9 implementation)
- **Phase 4 (US2 - Mapping Validation)**: 11 tasks (4 tests + 7 implementation)
- **Phase 5 (US4 - Shape Mismatch)**: 11 tasks (4 tests + 7 implementation)
- **Phase 6 (US3 - Batch Conversion)**: 10 tasks (4 tests + 6 implementation)
- **Phase 7 (CLI Interface)**: 8 tasks
- **Phase 8 (Polish)**: 13 tasks

**Parallel Opportunities**: 51 tasks marked [P] (65% of total)

**Independent Test Criteria**:
- **US1**: Convert .pdparams → .pth, load into PyTorch, verify parameters
- **US2**: Export mapping JSON, verify transformations and unmapped lists
- **US4**: Test strict mode (fails on mismatch) and permissive mode (skips with warning)
- **US3**: Batch convert 3 files, verify all outputs generated

**Suggested MVP Scope**: Phase 1-2 + User Story 1 only (T001-T025, 25 tasks)

---

## Notes

- [P] tasks = different files, no dependencies, can run in parallel
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing (validation-driven development per constitution)
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- All file paths are based on single project structure from plan.md
- Tests follow constitution Principle III: validation-driven development with 1e-5 tolerance
- Performance targets from spec: <2 min conversion (SC-001), ≤2x memory (SC-006)
