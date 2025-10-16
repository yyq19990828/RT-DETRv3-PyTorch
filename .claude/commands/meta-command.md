---
description: Generate custom Claude Code slash commands for the current project
allowed-tools: Read(**), Write(.claude/commands/**/*.md), Glob(**), Grep(**), WebFetch(docs.claude.com)
---

## Usage

`/meta-command $ARGUMENTS`

## Objective

Automatically generate custom Claude Code slash commands(SHOULD USE ENGLISH TO SAVE TOKENS) based on project needs, following the standard command template format with complete frontmatter metadata, usage instructions, execution steps, and security boundaries.

## Execution Steps

**Step 1**: Parse and validate $ARGUMENTS
- If `$ARGUMENTS` is empty, stop execution and provide usage example as below:
  ```
  Usage: /meta-command $ARGUMENTS

  Example: /meta-command "Create a command to run tests and generate coverage report, needs Bash and Read tools"
  Example: /meta-command "Build a deployment command that checks git status, runs build, and creates release notes"
  ```
- Parse `$ARGUMENTS` to extract:
  - Command name (inferred from description)
  - Command description
  - Required tools and permissions
  - Expected functionality
- **[important]** Intelligent inference with confirmation mode:
  - Parse `$ARGUMENTS` to identify keywords that suggest specific tools (e.g., "git" → git tools, "test" → testing tools, "perplexity" → MCP perplexity tools)
  - Automatically infer required tools based on description keywords
  - Present a complete command specification including:
    * Inferred command name
    * Identified required tools and permissions
    * Proposed functionality scope
  - Ask user: "Does this specification match your requirements? Type 'yes' to proceed, or provide corrections."
- If user provides corrections, incorporate them and re-present the specification
- Once confirmed, echo: `Generating command definition file...`

**Step 2**: Fetch command build docs
- Use WebFetch to retrieve latest documentation from `https://docs.claude.com/en/docs/claude-code/slash-commands#custom-slash-commands`
- Scan `.claude/commands/` directory to understand existing command patterns
- Analyze project type and tech stack characteristics
- Identify common operation patterns and tool requirements

**Step 3**: Generate command specification
- Infer command name from the description in `$ARGUMENTS`
- Determine required tool permissions (Bash, Read, Write, Edit, Glob, Grep, etc.)
- Design parameter structure and usage patterns
- Define execution orchestration

**Step 4**: Create command file
- Generate standard frontmatter metadata (description, allowed-tools, argument-hint [optional - only if command takes specific arguments])
- Write detailed usage instructions and objective description
- Define clear execution steps and orchestration instructions
- Add security boundaries and output requirements

**Step 5**: Validate and optimize
- Check command syntax correctness
- Ensure tool permissions follow minimum privilege principle
- Verify consistency with existing commands

## Command Template Structure

Generated commands will include this standard structure:
```markdown
---
description: [Specific functionality description]
allowed-tools: [Minimal privilege tool set]
argument-hint: [Optional - only if command takes specific arguments]
---

## Usage
`/command-name $ARGUMENTS`

## Objective
## Execution Steps
## Security and Boundaries
## Output Requirements
```

## Security and Boundaries

- Only create command definition files, do not modify project source code
- Follow minimum privilege principle for tool access configuration
- Ensure generated commands comply with Claude Code specifications
- Avoid generating commands with potential security risks

## Output Requirements

If `$ARGUMENTS` is empty, output:
```
Usage: /meta-command $ARGUMENTS

Example: /meta-command "Create a command to run tests and generate coverage report, needs Bash and Read tools"
Example: /meta-command "Build a deployment command that checks git status, runs build, and creates release notes"
```

Otherwise, output:
- Path of newly created command file
- Command functionality overview and usage method
- Configured tool permissions explanation
- Suggested testing approach

If command already exists, ask whether to overwrite or create variant version.