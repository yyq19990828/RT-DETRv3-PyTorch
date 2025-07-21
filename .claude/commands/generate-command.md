---
allowed-tools: Read(**), Write(**), Edit(**), Glob(**), Bash(mkdir:*), Bash(ls:*), Bash(cat:*), WebFetch(https://docs.anthropic.com/zh-CN/docs/claude-code/slash-commands)
description: 生成自定义Claude Code斜杠命令文件
---

# 生成自定义命令

## 功能描述
根据用户输入的需求或文件内容，生成符合Claude Code规范的自定义斜杠命令文件。

## 参数说明
`$ARGUMENTS` 可以是以下任一类型：
- **文件路径**: 读取指定文件内容作为命令需求
- **命令描述**: 直接描述要创建的命令功能
- **命令名称**: 只提供命令名称，后续交互确定具体需求

## 使用示例
```
/project:generate-command create-api-endpoint
/project:generate-command ./requirements.txt
/project:generate-command "创建一个用于代码审查的命令"
```

## 执行流程

### 1. 参数分析和需求确定
首先分析 `$ARGUMENTS` 的类型和内容：

```
如果 $ARGUMENTS 是一个不包含空格的字符串，且 (以'/' 或 './' 开头或包含文件扩展名):
  - 作为文件路径处理
  - 使用 Read 工具读取文件内容
  - 根据文件内容推断命令需求
否则 (即使参数中包含文件名，例如 "为 a.py 创建一个 lint 命令"):
  - 作为命令描述或名称处理
  - 直接解析用户意图
```

### 2. 命令需求分析
根据输入内容确定：
- 命令的核心功能
- 所需的工具权限
- 参数接收方式
- 输出格式要求

### 3. 生成命令文件
使用标准的Claude Code命令格式：

```markdown
---
allowed-tools: [根据功能需求确定的工具列表]
description: [简洁的命令描述]
---

# [命令标题]

## 功能描述
[详细的功能说明]

## 参数说明
$ARGUMENTS 的使用方式和格式

## 使用示例
具体的使用案例

## 执行步骤
详细的执行逻辑和流程

## 输出格式
期望的输出结果格式
```

### 4. 工具权限配置
根据命令功能需求，在 `allowed-tools` 中配置合适的工具权限：
- 文件操作：`Read(**)`、`Write(**)`、`Edit(**)`
- 搜索功能：`Glob(**)`、`Grep(**)`
- 命令执行：`Bash(command:*)`
- 网络请求：`WebFetch(url)`

### 5. 质量验证
确保生成的命令文件：
- [ ] YAML元数据格式正确
- [ ] 工具权限配置合理
- [ ] 参数处理逻辑清晰
- [ ] 执行步骤可操作
- [ ] 错误处理机制完善
- [ ] 用户体验友好

### 6. 文件保存
将生成的命令保存到 `.claude/commands/` 目录下，文件名格式为 `{command-name}.md`

## 参数处理示例

### 处理文件路径参数
```
输入: /project:generate-command ./api-spec.yaml
处理: 
1. 检测到文件路径格式
2. 读取 api-spec.yaml 内容
3. 根据API规范生成相应的命令
```

### 处理描述性参数
```
输入: /project:generate-command "创建代码审查命令"
处理:
1. 解析功能描述
2. 确定所需工具和权限
3. 生成对应的命令结构
```

### 处理命令名称参数
```
输入: /project:generate-command test-runner
处理:
1. 识别为命令名称
2. 交互式询问具体需求
3. 根据用户反馈生成命令
```

## 最佳实践

1. **权限最小化**: 只配置命令实际需要的工具权限
2. **错误处理**: 包含参数验证和异常情况处理
3. **用户友好**: 提供清晰的使用说明和示例
4. **可扩展性**: 设计支持参数扩展的结构
5. **文档完整**: 确保命令功能和使用方法说明充分

记住：生成的命令应该符合Claude Code的规范，具有良好的用户体验和可靠的执行逻辑。