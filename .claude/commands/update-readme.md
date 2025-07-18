---
allowed-tools: [Bash, Read, Write]
description: 根据已暂存的 git 修改, 自动更新项目的自述文件 (README.md)。
---

# 更新 README.md

## 功能描述
本命令会分析 Git 暂存区中的代码变更，并结合项目现有的 `README.md` 文件，智能地生成更新日志或功能说明，然后更新 `README.md` 文件。

## 执行步骤
1.  **获取代码变更**:
    使用 `git diff --staged` 命令，获取所有已暂存的文件变更内容。

2.  **读取当前 README**:
    使用 `Read` 工具读取 `README.md` 文件的现有内容。

3.  **生成更新内容**:
    结合第 1 步获取的变更和第 2 步的 `README.md` 内容，分析并生成一段简洁、准确的更新描述。通常会添加在 "更新日志" 或类似的章节下。

4.  **更新 README 文件**:
    使用 `Write` 工具将包含新描述的完整内容写回到 `README.md` 文件。

## 使用示例
当您完成了一些功能开发并使用 `git add` 将变更暂存后，可以直接运行此命令来自动化文档更新。

`/update-readme`
