---
description: Ask questions using Perplexity with automatic README.md context integration
allowed-tools: Read(**), mcp__mcpm_perplexity__perplexity_ask
argument-hint: [question]
---

## Usage

`/perplexity-question $ARGUMENTS`

## Objective

Use Perplexity MCP tools to answer questions, automatically including project README.md as contextual information when the question is related to the current project.

## Execution Steps

**Step 1**: Validate input
- Check if `$ARGUMENTS` contains a question
- If no question provided, prompt user to provide a question

**Step 2**: Read project context
- Check if README.md exists in the current project
- If README.md exists, read its content to understand project context
- Determine if the question is related to the current project

**Step 3**: Prepare context for Perplexity
- Format the user's question
- If question appears project-related and README.md exists:
  - Include README.md content as supplementary context
  - Create a comprehensive context message combining the question and project info

**Step 4**: Query Perplexity
- Use `mcp__mcpm_perplexity__perplexity_ask` to get the answer
- Structure the conversation with appropriate system/user messages
- Include project context when relevant

**Step 5**: Present results
- Display the Perplexity response
- Indicate if project context was included in the query

## Security and Boundaries

- Only read README.md from current project directory
- Do not include sensitive project information in external queries
- Respect user privacy by only using publicly shareable project context
- Limit context to README.md content only

## Output Requirements

- Clear, formatted response from Perplexity
- Indication of whether project context was included
- If README.md was used, mention it was included as context
- Handle cases where README.md doesn't exist gracefully