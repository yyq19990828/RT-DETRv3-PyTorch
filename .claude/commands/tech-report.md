---
description: Generate technical analysis report from academic paper and its implementation code
allowed-tools: Bash, Read, Glob, Grep, Write, Edit, Update mcp__mcpm_perplexity__perplexity_ask, mcp__mcpm_markitdown__convert_to_markdown
argument-hint: [path]
---

## User Input

```text
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty).

## Outline

1. **Setup**: Run `.claude/scripts/setup-tech-report.sh --json "$ARGUMENTS"` from repo root and parse JSON for TARGET_PATH, REPORT_PATH, PDF_PATH, PDF_COUNT, HAS_CODE, TEMPLATE. For single quotes in args like "I'm Groot", use escape syntax: e.g 'I'\''m Groot' (or double-quote if possible: "I'm Groot").

2. **PDF Identification**:
   - If PDF_COUNT=1: Use PDF_PATH automatically
   - If PDF_COUNT>1: List all PDFs, ask user to select one
   - If PDF_COUNT=0: Ask user to provide PDF path
   - Validate PDF exists and is readable

3. **Load context & Parse template**:
   - Read PDF using Read tool (supports PDF natively). If Read fails, try `mcp__mcpm_markitdown__convert_to_markdown`
   - Read REPORT_PATH template (already copied by setup script)
   - **Extract template section structure** by finding all `## ` and `### ` headings to determine edit boundaries

4. **Execute analysis workflow** (Phases 1-3): Collect data from paper and code:
   - Extract paper overview (title, authors, abstract)
   - Identify key algorithms and equations
   - Map each algorithm to code implementation with exact line numbers
   - Use LaTeX for all formulas with `\text{}` for non-math text
   - Create one-to-one correspondence tables
   - Assess code quality and gaps

5. **Generate report incrementally** (Phase 4): Use Edit tool to replace each template section sequentially (see Phase 4 details below)

6. **Stop and report**: Output REPORT_PATH location and summary statistics.

## Key Analysis Steps

### Phase 1: Paper Analysis

1. **Extract paper content**:
   - Title, authors, publication info
   - Abstract (condense to 3-5 key points)
   - Main contributions
   - Core algorithms (identify 3-5 major ones)
   - Key equations (identify 5-10 important formulas)

2. **Organize by sections**:
   - Methodology sections (algorithm descriptions)
   - Mathematical framework (equations)
   - Experimental setup (if relevant)

### Phase 2: Code Analysis

1. **Discover code structure**:
   ```bash
   # Use Glob to find source files
   **/*.py (or *.js, *.cpp, etc.)
   ```

2. **Identify entry points**:
   - Main scripts (train.py, main.py, etc.)
   - Core model files
   - Training/inference pipelines

3. **Map algorithms to code**:
   - For each algorithm in paper:
     - Read relevant code files completely
     - Find exact line numbers where implemented
     - Extract actual code snippets (not summaries)
     - Document format: `file.py:line_start-line_end`

### Phase 3: Cross-Reference Mapping

**CRITICAL**: Create detailed one-to-one mappings with exact line numbers.

For each algorithm:
1. **Paper**: Quote key concept, write formula in LaTeX
2. **Code**: Show actual implementation with line numbers
3. **Mapping**: Create correspondence table

Example format:
```markdown
#### Algorithm: Hybrid Encoder

**Paper** (Section 4.2):
> "We design an efficient hybrid encoder..."

**Formula**:
$$
\mathcal{O} = \text{CCFF}(\text{AIFI}(S_5), S_4, S_3)
$$

**Code** (`hybrid_encoder.py:283-322`):
```python
def forward(self, feats):
    # Line 293: AIFI only on S5
    memory = self.encoder(proj_feats[2].flatten(2))
    # Line 304: CCFF top-down fusion
    for idx in range(len(feats)-1, 0, -1):
        inner_out = self.fpn_blocks[idx](...)
    return outs
```

**Correspondence**:

| Paper | Code | Line |
|-------|------|------|
| AIFI(S₅) | `self.encoder()` | 299 |
| CCFF fusion | `fpn_blocks` | 311 |
```

### Phase 4: Generate Report (Incremental Section-by-Section)

**IMPORTANT**: Use **Edit tool** to incrementally replace each template section, NOT Write tool.

**Workflow**:
1. **Parse template structure** from REPORT_PATH to identify all section markers (## headings)
2. **Generate content for each section** sequentially:
   - Read current section from template
   - Generate complete content based on collected data
   - Use `Edit` tool to replace ONLY that section (match exact heading + placeholder content)
   - Preserve all other sections intact

**Section Order** (follow template structure):
1. `## Paper Overview` → Replace [PAPER_TITLE], [Extract from paper], etc.
2. `## Abstract Summary` → Replace [Condensed abstract with key points]
3. `## Methodology Analysis` → Replace each `#### [Algorithm Name from Paper]` subsection
4. `### Mathematical Framework` → Replace each `#### Equation X` subsection
5. `## Implementation Analysis` → Replace code structure, entry points
6. `### Algorithm Implementation` → Replace each algorithm mapping
7. `## Paper-to-Code Correspondence` → Fill correspondence table
8. `## Code Quality Assessment` → Replace [Clear Design], [Comments], etc.
9. `## Implementation Gaps` → Replace [Feature X] items
10. `## Potential Improvements` → Add concrete examples
11. `## Reproducibility Notes` → Fill environment, training, eval commands
12. `## Summary Statistics` → Replace [X], [Y], [Z] with actual numbers

**Edit Strategy**:

**Step 1: Parse Template Sections**
```python
# Read template to extract section boundaries
template_content = Read(REPORT_PATH)
sections = []  # List of (heading, start_line, end_line, placeholder_text)

# Identify sections by ## headings and their content until next ## or end
# This determines exact old_string for each Edit call
```

**Step 2: Generate Each Section Content**
For each section in order:
1. Generate complete content based on collected data (paper + code analysis)
2. Format with proper LaTeX, code blocks, and tables
3. Use Edit tool with exact match from template

**Edit Pattern Examples**:

<details>
<summary>Example 1: Paper Overview</summary>

```python
Edit(
    file_path=REPORT_PATH,
    old_string="""## Paper Overview

- **Title**: [Extract from paper]
- **Authors**: [Extract from paper]
- **Publication**: [Conference/Journal, Year]
- **Core Contribution**: [Main innovation in 1-2 sentences]
- **Key Concepts**: [List 3-5 key technical concepts]""",
    new_string="""## Paper Overview

- **Title**: DETRs Beat YOLOs on Real-time Object Detection
- **Authors**: Yian Zhao, Wenyu Lv, et al.
- **Publication**: arXiv:2304.08069v3 [cs.CV] 3 Apr 2024
- **Core Contribution**: First real-time end-to-end detector achieving 53.1% AP @ 108 FPS
- **Key Concepts**:
  1. Efficient Hybrid Encoder (AIFI + CCFF)
  2. Uncertainty-minimal Query Selection
  3. Multi-scale Deformable Attention"""
)
```
</details>

<details>
<summary>Example 2: Algorithm Section</summary>

```python
Edit(
    file_path=REPORT_PATH,
    old_string="""#### 1. [Algorithm Name from Paper]

**Paper Description** (Section X.X):
> [Quote or paraphrase key concept from paper]

**Key Innovation**:
[What makes this approach novel compared to prior work]

**Mathematical Formulation**:
$$
\text{[Description]}: \mathcal{F}(x) = \text{[formula]}
$$

Where:
- $x$: [description]
- $\mathcal{F}$: [description]

**Code Implementation**:
```python
# File: path/to/file.py:line_start-line_end
# Function/Class: ExactName

def function_name(param1, param2):
    """[Docstring if present]"""
    # Line X: implements equation Y from paper
    result = actual_implementation
    return result
```

**Correspondence Notes**:
[Explain how the code realizes the theoretical concept, note any deviations]""",
    new_string="""#### 1. Efficient Hybrid Encoder

**Paper Description** (Section 4.2):
> "Design an efficient hybrid encoder to expeditiously process multi-scale features by decoupling intra-scale interaction and cross-scale fusion"

**Key Innovation**:
Decouples AIFI (attention on S5 only) and CCFF (CNN-based fusion) to reduce computation by 35% while improving accuracy by 0.4% AP

**Mathematical Formulation**:
$$
\begin{aligned}
Q = K = V &= \text{Flatten}(S_5) \\
F_5 &= \text{Reshape}(\text{AIFI}(Q, K, V)) \\
\mathcal{O} &= \text{CCFF}(\{S_3, S_4, F_5\})
\end{aligned}
$$

Where:
- $S_3, S_4, S_5$: Multi-scale features at strides 8, 16, 32
- $\text{AIFI}$: Attention-based intra-scale interaction on S5
- $\text{CCFF}$: CNN-based cross-scale fusion (FPN + PAN)

**Code Implementation**:
```python
# File: rtdetr_pytorch/src/zoo/rtdetr/hybrid_encoder.py:283-322
# Class: HybridEncoder

def forward(self, feats):
    # Line 285: Channel projection
    proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

    # Line 288-300: AIFI - Attention on S5 only
    for i, enc_ind in enumerate(self.use_encoder_idx):  # [2] for S5
        src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
        memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
        proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(...)

    # Line 304-320: CCFF - FPN + PAN fusion
    inner_outs = [proj_feats[-1]]
    for idx in range(len(self.in_channels) - 1, 0, -1):
        # Top-down FPN
        inner_out = self.fpn_blocks[...](torch.concat([upsample_feat, feat_low], dim=1))
        inner_outs.insert(0, inner_out)

    # Bottom-up PAN
    outs = [inner_outs[0]]
    for idx in range(len(self.in_channels) - 1):
        out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_high], dim=1))
        outs.append(out)

    return outs
```

**Correspondence Notes**:
- AIFI operates only on S5 (line 289: `use_encoder_idx=[2]`), matching paper's DS5 variant
- CCFF implements FPN (lines 304-312) and PAN (lines 314-320) exactly as described
- RepVggBlock used in fusion blocks (line 239: `CSPRepLayer`) for efficient inference"""
)
```
</details>

**Critical Rules**:
- **Match exact template text** in `old_string` including whitespace, brackets, placeholders
- Use LaTeX for all formulas: `$$\text{Loss} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{box}}$$`
- Include exact file paths with line numbers: `file.py:100-150`
- Show actual code snippets (not pseudocode)
- Create comprehensive correspondence tables
- Preserve document structure by editing sections in order
- If Edit fails due to mismatch, read current section content and retry with exact match

**Error Handling**:
- If `old_string` not found: Read REPORT_PATH again to get current content, extract exact section text
- If section was already modified: Skip with message "Section already filled"
- If multiple algorithms found: Create separate `#### N.` subsections for each
- If no code found for algorithm: Note in "Correspondence Notes" with "⚠ Missing"

**Progress Tracking** (use TodoWrite):
Create todo list with all sections to track progress:
```
1. [in_progress] Filling Paper Overview
2. [pending] Filling Abstract Summary
3. [pending] Filling Methodology Analysis
...
12. [pending] Filling Summary Statistics
```
Mark each section as completed after successful Edit.

## Formatting Rules

**CRITICAL REQUIREMENTS**:

1. **LaTeX Formulas**:
   - Use `$$...$$` for display math
   - Use `$...$` for inline math
   - Wrap non-math text: `\text{encoder features}`
   - Example: `$$U(\hat{X}) = \|P(\hat{X}) - C(\hat{X})\|$$`

2. **Code Locations**:
   - Always include file path and line range: `file.py:100-150`
   - Show actual code, not summaries
   - Add line-by-line comments for correspondence

3. **One-to-One Mapping**:
   - Each paper algorithm → specific code location
   - Each equation → implementation snippet
   - Use correspondence tables for clarity

4. **Variable Mapping**:
   - Paper notation → code variable name
   - Include type/shape information
   - Example: `$\hat{X}$ → `features` ([B, 256, H, W])`

## Output Requirements

**Standard output**:
```
Analyzing directory: /path/to/project
Found paper: paper.pdf
Extracting paper content...
  - Title: [...]
  - Algorithms: 5
  - Equations: 8
Analyzing source code...
  - Python files: 40
  - Entry point: tools/train.py
  - Core modules: 10
Generating report...

✓ Technical report generated: /path/to/project/tech-report.md

Summary:
- Source files analyzed: 40
- Paper sections mapped: 10
- Implementation completeness: 95%
- Code quality: Good
- Report location: /absolute/path/tech-report.md
```

**Error cases**:
- "Multiple PDFs found. Please select one:"
- "No PDF found. Please provide paper path:"
- "Cannot read PDF file. Please check file permissions."
- "No source code found in directory."

## Context

Analysis target: $ARGUMENTS (or current directory if empty)
