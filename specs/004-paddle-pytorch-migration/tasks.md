# Tasks: RT-DETRv3 Paddle to PyTorch Migration Completion

**Input**: Design documents from `/specs/004-paddle-pytorch-migration/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: Based on Constitution Principle III (Validation-Driven Development), tests are MANDATORY for this feature

**Organization**: Tasks grouped by user story to enable independent implementation and testing

## Format: `[ID] [P?] [Story] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: User story label (US1, US2, US3, US4, US5)
- Include exact file paths in descriptions

## Path Conventions
- Single project structure: `rtdetrv3_pytorch/` at repository root
- Tests in: `rtdetrv3_pytorch/tests/`, root-level validation scripts
- Configs in: `rtdetrv3_pytorch/configs/` or `configs/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Validate existing project structure and prepare for migration

- [ ] T001 Verify project structure matches plan.md specifications
- [ ] T002 Confirm all 8 core components exist (RTDETRv3, ResNet, HybridEncoder, RTDETRTransformerv3, DINOv3Head, PPYOLOEHead, DINOv3Loss, 1 additional)
- [ ] T003 [P] Verify pytest environment configured with markers (unit, integration, numerical)
- [ ] T004 [P] Confirm PyYAML ≥6.0 installed for config parsing

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Registry system enhancements that ALL user stories depend on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T005 Verify existing Registry class in rtdetrv3_pytorch/models/__init__.py supports __inject__, __shared__, __category__
- [ ] T006 [P] Verify global create() function exists in rtdetrv3_pytorch/models/__init__.py
- [ ] T007 [P] Verify all 6 registry instances exist (ARCHITECTURE_REGISTRY, BACKBONE_REGISTRY, NECK_REGISTRY, TRANSFORMER_REGISTRY, HEAD_REGISTRY, LOSS_REGISTRY)
- [ ] T008 Verify Registry.create() method properly calls from_config() if defined (FR-007)
- [ ] T009 Add validation helper for component protocol compliance in rtdetrv3_pytorch/models/__init__.py

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Complete Component Registration System (Priority: P1) 🎯 MVP

**Goal**: Ensure all 8 core components properly registered to enable PaddlePaddle-style instantiation

**Independent Test**: Run `verify_paddle_migration.py` to confirm all components registered

### Tests for User Story 1

**NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T010 [P] [US1] Write unit test for BACKBONE_REGISTRY.list() in rtdetrv3_pytorch/tests/unit/test_registry.py
- [ ] T011 [P] [US1] Write unit test for NECK_REGISTRY.list() in rtdetrv3_pytorch/tests/unit/test_registry.py
- [ ] T012 [P] [US1] Write unit test for TRANSFORMER_REGISTRY.list() in rtdetrv3_pytorch/tests/unit/test_registry.py
- [ ] T013 [P] [US1] Write unit test for HEAD_REGISTRY.list() in rtdetrv3_pytorch/tests/unit/test_registry.py
- [ ] T014 [P] [US1] Write unit test for LOSS_REGISTRY.list() in rtdetrv3_pytorch/tests/unit/test_registry.py
- [ ] T015 [P] [US1] Write unit test for ARCHITECTURE_REGISTRY.list() in rtdetrv3_pytorch/tests/unit/test_registry.py
- [ ] T016 [US1] Write integration test for component registration on module import in rtdetrv3_pytorch/tests/integration/test_registration.py

### Implementation for User Story 1

- [ ] T017 [P] [US1] Add @BACKBONE_REGISTRY.register() and __category__ to ResNet in rtdetrv3_pytorch/models/backbones/resnet.py
- [ ] T018 [P] [US1] Import ResNet in rtdetrv3_pytorch/models/backbones/__init__.py to trigger registration
- [ ] T019 [P] [US1] Add @NECK_REGISTRY.register() and __category__ to HybridEncoder in rtdetrv3_pytorch/models/necks/hybrid_encoder.py
- [ ] T020 [P] [US1] Import HybridEncoder in rtdetrv3_pytorch/models/necks/__init__.py to trigger registration
- [ ] T021 [P] [US1] Add @TRANSFORMER_REGISTRY.register() and __category__ to RTDETRTransformerv3 in rtdetrv3_pytorch/models/transformers/rtdetr_transformer.py
- [ ] T022 [P] [US1] Import RTDETRTransformerv3 in rtdetrv3_pytorch/models/transformers/__init__.py to trigger registration
- [ ] T023 [P] [US1] Add @HEAD_REGISTRY.register() and __category__ to DINOv3Head in rtdetrv3_pytorch/models/heads/detr_head.py
- [ ] T024 [P] [US1] Add @HEAD_REGISTRY.register() and __category__ to PPYOLOEHead in rtdetrv3_pytorch/models/heads/ppyoloe_head.py
- [ ] T025 [US1] Import both heads in rtdetrv3_pytorch/models/heads/__init__.py to trigger registration
- [ ] T026 [P] [US1] Add @LOSS_REGISTRY.register() and __category__ to DINOv3Loss in rtdetrv3_pytorch/models/losses/detr_loss.py
- [ ] T027 [US1] Explicitly import DINOv3Loss in rtdetrv3_pytorch/models/losses/__init__.py to trigger registration
- [ ] T028 [P] [US1] Add @ARCHITECTURE_REGISTRY.register() and __category__ to RTDETRv3 in rtdetrv3_pytorch/models/rtdetrv3.py
- [ ] T029 [US1] Enhance verify_paddle_migration.py to report registration status for all 8 components
- [ ] T030 [US1] Run all US1 tests and verify they PASS (SC-001: all 8 components registered)

**Checkpoint**: All 8 components registered and discoverable via REGISTRY.list()

---

## Phase 4: User Story 2 - Implement Dependency Injection Chain (Priority: P1)

**Goal**: Enable automatic parameter passing from backbone → neck → transformer → head

**Independent Test**: Call `RTDETRv3.from_config(config)` and verify components receive upstream attributes

### Tests for User Story 2

- [ ] T031 [P] [US2] Write unit test for ResNet.out_shape attribute in rtdetrv3_pytorch/tests/unit/test_backbone.py
- [ ] T032 [P] [US2] Write integration test for backbone → neck injection in rtdetrv3_pytorch/tests/integration/test_dependency_injection.py
- [ ] T033 [P] [US2] Write integration test for neck → transformer injection in rtdetrv3_pytorch/tests/integration/test_dependency_injection.py
- [ ] T034 [P] [US2] Write integration test for transformer → head injection in rtdetrv3_pytorch/tests/integration/test_dependency_injection.py
- [ ] T035 [US2] Write end-to-end test for full dependency chain in rtdetrv3_pytorch/tests/integration/test_dependency_injection.py

### Implementation for User Story 2

- [ ] T036 [P] [US2] Add __inject__ = [] and __shared__ = [] to ResNet in rtdetrv3_pytorch/models/backbones/resnet.py
- [ ] T037 [US2] Verify ResNet._setup_out_shape() provides out_shape attribute (FR-010)
- [ ] T038 [P] [US2] Add __inject__ = [] and __shared__ = [] to HybridEncoder in rtdetrv3_pytorch/models/necks/hybrid_encoder.py
- [ ] T039 [P] [US2] Add __inject__ = [] and __shared__ = [] to RTDETRTransformerv3 in rtdetrv3_pytorch/models/transformers/rtdetr_transformer.py
- [ ] T040 [P] [US2] Add __inject__ = [] and __shared__ = ['num_classes', 'hidden_dim'] to DINOv3Head in rtdetrv3_pytorch/models/heads/detr_head.py
- [ ] T041 [P] [US2] Add __inject__ = [] and __shared__ = ['num_classes'] to PPYOLOEHead in rtdetrv3_pytorch/models/heads/ppyoloe_head.py
- [ ] T042 [US2] Add __inject__ = ['backbone', 'neck', 'transformer', 'detr_head'] to RTDETRv3 in rtdetrv3_pytorch/models/rtdetrv3.py
- [ ] T043 [US2] Add __shared__ = ['num_classes'] to RTDETRv3 in rtdetrv3_pytorch/models/rtdetrv3.py
- [ ] T044 [US2] Implement RTDETRv3.from_config() class method with dependency injection chain (FR-005, FR-006)
- [ ] T045 [US2] In from_config(), create backbone and inject out_shape to neck in rtdetrv3_pytorch/models/rtdetrv3.py
- [ ] T046 [US2] In from_config(), create neck and inject output to transformer in rtdetrv3_pytorch/models/rtdetrv3.py
- [ ] T047 [US2] In from_config(), create transformer and inject hidden_dim to head in rtdetrv3_pytorch/models/rtdetrv3.py
- [ ] T048 [US2] Run all US2 tests and verify dependency injection works (SC-003)

**Checkpoint**: Dependency injection chain fully functional

---

## Phase 5: User Story 3 - Enable Config-Driven Model Building (Priority: P2)

**Goal**: Allow building RT-DETRv3 from YAML/dict config without manual instantiation

**Independent Test**: Load example YAML config and call `create('RTDETRv3', global_config=config)` to build model

### Tests for User Story 3

- [ ] T049 [P] [US3] Write unit test for Registry.create() with nested config in rtdetrv3_pytorch/tests/unit/test_registry.py
- [ ] T050 [P] [US3] Write integration test for YAML config loading in rtdetrv3_pytorch/tests/integration/test_config_driven_build.py
- [ ] T051 [P] [US3] Write integration test for global_config parameter resolution in rtdetrv3_pytorch/tests/integration/test_config_driven_build.py
- [ ] T052 [US3] Write end-to-end test for complete model creation from config in rtdetrv3_pytorch/tests/integration/test_config_driven_build.py

### Implementation for User Story 3

- [ ] T053 [P] [US3] Create example PaddlePaddle-style YAML config in rtdetrv3_pytorch/configs/examples/rtdetrv3_r50_paddle_style.yml
- [ ] T054 [US3] Verify build_from_config() function handles 'type' key correctly (FR-012)
- [ ] T055 [US3] Verify global create() function searches all registries (FR-008)
- [ ] T056 [US3] Test parameter resolution priority (explicit > shared > component_cfg > default)
- [ ] T057 [US3] Add config validation helper to check for missing 'type' keys
- [ ] T058 [US3] Test error handling when component type not found (FR-013)
- [ ] T059 [US3] Run all US3 tests and verify config-driven building works (SC-002: 60% code reduction)

**Checkpoint**: Model can be built from YAML/dict config

---

## Phase 6: User Story 4 - Maintain Backward Compatibility (Priority: P2)

**Goal**: Ensure existing direct instantiation code continues to work unchanged

**Independent Test**: Run existing test suite to verify no regressions

### Tests for User Story 4

- [ ] T060 [P] [US4] Write numerical equivalence test for ResNet (direct vs registered) in rtdetrv3_pytorch/tests/numerical/test_registered_components.py
- [ ] T061 [P] [US4] Write numerical equivalence test for HybridEncoder (direct vs registered) in rtdetrv3_pytorch/tests/numerical/test_registered_components.py
- [ ] T062 [P] [US4] Write numerical equivalence test for RTDETRTransformerv3 (direct vs registered) in rtdetrv3_pytorch/tests/numerical/test_registered_components.py
- [ ] T063 [P] [US4] Write numerical equivalence test for DINOv3Head (direct vs registered) in rtdetrv3_pytorch/tests/numerical/test_registered_components.py
- [ ] T064 [P] [US4] Write numerical equivalence test for RTDETRv3 (direct vs registered) in rtdetrv3_pytorch/tests/numerical/test_registered_components.py
- [ ] T065 [US4] Write integration test for existing usage patterns in rtdetrv3_pytorch/tests/integration/test_backward_compat.py

### Implementation for User Story 4

- [ ] T066 [US4] Verify @register decorator is non-invasive (returns class unchanged)
- [ ] T067 [US4] Verify direct instantiation still works for all components (FR-009)
- [ ] T068 [US4] Verify no warnings/errors when using direct instantiation
- [ ] T069 [US4] Run existing rtdetrv3_pytorch/tests/unit/ tests to ensure no regressions
- [ ] T070 [US4] Run existing rtdetrv3_pytorch/tests/integration/ tests to ensure no regressions
- [ ] T071 [US4] Run all US4 numerical tests with tolerance <1e-5 (SC-004: 100% compatibility)

**Checkpoint**: All existing code continues to work without modification

---

## Phase 7: User Story 5 - Add Comprehensive Validation Tools (Priority: P3)

**Goal**: Provide tools to verify migration completeness and correctness

**Independent Test**: Run verification script and confirm all checks pass

### Tests for User Story 5

- [ ] T072 [P] [US5] Write unit test for migration validator in test_registry_system.py
- [ ] T073 [US5] Write test for component metadata validation in test_registry_system.py

### Implementation for User Story 5

- [ ] T074 [P] [US5] Enhance verify_paddle_migration.py to list all registered components
- [ ] T075 [P] [US5] Add check for missing __category__ annotations in verify_paddle_migration.py
- [ ] T076 [P] [US5] Add check for missing __inject__ annotations in verify_paddle_migration.py
- [ ] T077 [US5] Add dependency injection chain validator in verify_paddle_migration.py
- [ ] T078 [US5] Add performance benchmark (registry lookup <5ms) in verify_paddle_migration.py
- [ ] T079 [US5] Enhance test_registry_system.py to test from_config() for all components
- [ ] T080 [US5] Add validation for parameter resolution order
- [ ] T081 [US5] Run verification script and confirm SC-005 (completes in <2s)

**Checkpoint**: All validation tools working and reporting correct status

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Documentation and final validation

- [ ] T082 [P] Update API documentation based on contracts/registry-api.md
- [ ] T083 [P] Update component migration guide based on contracts/component-protocol.md
- [ ] T084 [P] Add usage examples to main README.md
- [ ] T085 Verify SC-006: Code structure 100% matches PaddlePaddle patterns
- [ ] T086 Verify SC-007: Documentation 100% complete (API reference, migration guide, examples)
- [ ] T087 Run complete test suite (unit + integration + numerical)
- [ ] T088 Generate test coverage report (aim for >90%)
- [ ] T089 Run performance benchmark on all components
- [ ] T090 Final verification: Run quickstart.md validation for all 5 user stories

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup - BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational completion
- **User Story 2 (Phase 4)**: Depends on Foundational completion + US1 (needs registered components)
- **User Story 3 (Phase 5)**: Depends on Foundational completion + US1 + US2 (needs injection chain)
- **User Story 4 (Phase 6)**: Depends on US1 + US2 + US3 (validates all features)
- **User Story 5 (Phase 7)**: Depends on US1-4 (validates complete migration)
- **Polish (Phase 8)**: Depends on all user stories

### User Story Dependencies

- **US1 (Component Registration)**: Foundation only - can start after Phase 2
- **US2 (Dependency Injection)**: Requires US1 (needs registered components)
- **US3 (Config-Driven Building)**: Requires US1 + US2 (needs registration + injection)
- **US4 (Backward Compatibility)**: Requires US1 + US2 + US3 (validates all features)
- **US5 (Validation Tools)**: Requires US1-4 (validates complete migration)

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Registration before injection
- Injection before config-driven building
- Core features before validation tools

### Parallel Opportunities

**Setup Phase**:
- T003 [P] and T004 [P] can run in parallel

**Foundational Phase**:
- T006 [P] and T007 [P] can run in parallel

**User Story 1**:
- Tests T010-T015 [P] can run in parallel (different files)
- Implementations T017 [P], T019 [P], T021 [P], T023-T024 [P], T026 [P], T028 [P] can run in parallel (different components)

**User Story 2**:
- Tests T031-T034 [P] can run in parallel
- Implementations T036-T041 [P] can run in parallel (different component files)

**User Story 3**:
- Tests T049-T051 [P] can run in parallel

**User Story 4**:
- Tests T060-T064 [P] can run in parallel (different components)

**User Story 5**:
- Implementations T074-T076 [P] can run in parallel (different checks)

**Polish Phase**:
- T082-T084 [P] can run in parallel (different docs)

---

## Parallel Example: User Story 1

```bash
# Launch all test writing tasks together:
Task: "Write unit test for BACKBONE_REGISTRY.list() in rtdetrv3_pytorch/tests/unit/test_registry.py"
Task: "Write unit test for NECK_REGISTRY.list() in rtdetrv3_pytorch/tests/unit/test_registry.py"
Task: "Write unit test for TRANSFORMER_REGISTRY.list() in rtdetrv3_pytorch/tests/unit/test_registry.py"
Task: "Write unit test for HEAD_REGISTRY.list() in rtdetrv3_pytorch/tests/unit/test_registry.py"
Task: "Write unit test for LOSS_REGISTRY.list() in rtdetrv3_pytorch/tests/unit/test_registry.py"
Task: "Write unit test for ARCHITECTURE_REGISTRY.list() in rtdetrv3_pytorch/tests/unit/test_registry.py"

# Launch all component registration tasks together:
Task: "Add @BACKBONE_REGISTRY.register() and __category__ to ResNet in rtdetrv3_pytorch/models/backbones/resnet.py"
Task: "Add @NECK_REGISTRY.register() and __category__ to HybridEncoder in rtdetrv3_pytorch/models/necks/hybrid_encoder.py"
Task: "Add @TRANSFORMER_REGISTRY.register() and __category__ to RTDETRTransformerv3 in rtdetrv3_pytorch/models/transformers/rtdetr_transformer.py"
Task: "Add @HEAD_REGISTRY.register() and __category__ to DINOv3Head in rtdetrv3_pytorch/models/heads/detr_head.py"
Task: "Add @HEAD_REGISTRY.register() and __category__ to PPYOLOEHead in rtdetrv3_pytorch/models/heads/ppyoloe_head.py"
Task: "Add @LOSS_REGISTRY.register() and __category__ to DINOv3Loss in rtdetrv3_pytorch/models/losses/detr_loss.py"
Task: "Add @ARCHITECTURE_REGISTRY.register() and __category__ to RTDETRv3 in rtdetrv3_pytorch/models/rtdetrv3.py"
```

---

## Implementation Strategy

### MVP First (User Stories 1 + 2)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL)
3. Complete Phase 3: User Story 1 (Component Registration)
4. Complete Phase 4: User Story 2 (Dependency Injection)
5. **STOP and VALIDATE**: Test US1 + US2 independently
6. Minimum viable system ready (can create models via registry + injection)

### Incremental Delivery

1. MVP: US1 + US2 (Core functionality)
2. Add US3: Config-driven building (Convenience feature)
3. Add US4: Backward compatibility validation (Safety net)
4. Add US5: Validation tools (Developer experience)
5. Polish: Documentation and final validation

### Timeline Estimate

Based on quickstart.md estimates:

- **Setup + Foundational**: 30 minutes (T001-T009)
- **US1 (Registration)**: 1.5 hours (8 components × 10 min each + tests)
- **US2 (Injection)**: 2 hours (complex from_config logic + tests)
- **US3 (Config-driven)**: 1 hour (YAML example + tests)
- **US4 (Backward compat)**: 1 hour (numerical tests)
- **US5 (Validation)**: 1 hour (tooling enhancements)
- **Polish**: 1 hour (docs + final validation)

**Total**: ~8 hours (1 developer day)

---

## Notes

- [P] tasks = different files, can run in parallel
- [Story] label maps task to user story for traceability
- Tests written FIRST (TDD approach required by Constitution)
- Each user story independently testable at checkpoint
- Commit after each task or logical group
- Numerical tolerance: <1e-5 for FP32 (Constitution requirement)
- Performance target: Registry overhead <5ms per component
- Success criteria validated at each checkpoint
