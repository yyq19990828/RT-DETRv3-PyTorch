# AGENTS.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 0. Project-Specific Rules

- Communicate with the user in Chinese.
- Prefer the repository's uv-managed `.venv` for Python commands and dependency work.
- Remove test caches, temporary checkpoints, build outputs, and other intermediate artifacts created by your work after validation. Keep `.venv` unless the user explicitly asks to remove it.
- Treat `third-party/RT-DETRv3-paddle` as a read-only reference submodule. Keep Paddle and migration-only dependencies in the `dev` extra; the core PyTorch runtime must not require Paddle imports.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Documentation Organization

- Put executable implementation plans in `docs/plans/`. Do not create or restore a top-level `specs/` directory.
- Use `docs/plans/TEMPLATE.md` for new plans and name standalone plans `YYYY-MM-DD-<topic>.md`. Record status, scope, dependencies, risks, validation criteria, and actual completion evidence.
- Keep the repository-wide unfinished migration outline in `ROADMAP.md`. Update it when verified work changes milestone status; do not duplicate its entire backlog in a new plan.
- Put reusable Paddle-to-PyTorch migration knowledge in `docs/migrations/`, including framework semantic comparisons, weight conversion rules, configuration and registry behavior, limitations, numerical validation methods, and troubleshooting.
- When a plan finishes, update its completion record and promote reusable findings to `docs/migrations/`. Do not leave important conclusions only in task checklists or chat logs.
- Historical documents must carry a dated snapshot notice and must not claim to represent the current repository state.
- Keep `docs/plans/README.md` and `docs/migrations/README.md` indexes current whenever files are added, moved, or removed.
- Use repository-relative links and paths in documentation. Never commit workstation-specific absolute paths.

## 6. Migration Evidence Rules

- Distinguish **verified**, **observed**, **inferred**, and **planned** statements. A historical checked box is not current evidence.
- Do not claim Paddle/PyTorch parity from matching class names, tensor shapes, successful imports, or deterministic output alone.
- Numerical parity requires the same checkpoint, preprocessing, inputs, evaluation mode, dtype, and clearly recorded tolerances. Compare the first divergent intermediate activation before debugging final predictions.
- Treat similar API names as hypotheses, not proof of equivalent semantics. Check optimizer equations, scheduler step units, DataLoader/collation behavior, padding, interpolation, random-number sources, BatchNorm state, and distributed reduction behavior.
- For checkpoint conversion, validate name mapping and tensor layout separately. Paddle Linear weights commonly require transposition; convolution weights usually do not. Always validate against a target `state_dict` when available.
- Record Python, Paddle, PyTorch, CUDA/cuDNN, device, model variant, dataset split, command, seed, dtype, and tolerance for reproducible migration results.
- Keep core-only tests independent of Paddle. Mark Paddle-dependent tests explicitly and run them from `uv sync --extra dev` environments.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
