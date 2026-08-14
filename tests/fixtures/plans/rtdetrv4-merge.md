# RT-DETRv4 Merge Validation Plan

Deterministic validation plan consumed by the `tools/dev` drivers and their
unit tests. The drivers only hash this file (normalizing task checkbox
state); receipt file names follow the `task-N-rtdetrv4-merge.json` pattern.

- [ ] 1. Verify checkpoint parity against the pinned upstream revision.
- [x] 2. Run reduced train/resume for the smallest variant.
- [ ] F1. Confirm eager and exported surface agreement.
- [x] F2. Record the final quality evidence receipts.
