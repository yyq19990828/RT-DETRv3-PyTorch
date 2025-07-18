---
allowed-tools: [Bash(command:*)]
description: 分析指定git哈希值的提交信息和修改文件
---

# Git提交分析命令

## 功能描述
根据提供的git哈希值，显示该提交的详细信息，包括提交消息、作者、时间戳以及所有修改的文件列表。

## 参数说明
$ARGUMENTS 应该是一个有效的git提交哈希值（完整或缩写形式）

## 使用示例
```
/commit-info 1901fdbb
/commit-info 4bb7765e
/commit-info 61d2bbb1f8a3c5d2e9f7b4a8c6d1e5f9g2h3i4j5
```

## 执行步骤

### 1. 参数验证
首先验证提供的哈希值是否有效：

```bash
git rev-parse --verify $ARGUMENTS^{commit}
```

### 2. 获取提交基本信息
显示提交的基本信息：

```bash
git show --no-patch --format="提交哈希: %H%n作者: %an <%ae>%n时间: %ad%n%n提交消息:%n%B" --date=format:"%Y-%m-%d %H:%M:%S" $ARGUMENTS
```

### 3. 获取修改文件列表
显示该提交中修改的所有文件：

```bash
git show --name-status $ARGUMENTS
```

### 4. 统计信息
显示修改的统计信息：

```bash
git show --stat $ARGUMENTS
```

## 输出格式
命令将按以下格式输出：

```
=== 提交信息 ===
提交哈希: [完整哈希值]
作者: [作者姓名] <[邮箱]>
时间: [提交时间]

提交消息:
[提交消息内容]

=== 修改文件 ===
[状态标识] [文件路径]
...

=== 统计信息 ===
[文件修改统计]
```

状态标识说明：
- M: 修改 (Modified)
- A: 新增 (Added)
- D: 删除 (Deleted)
- R: 重命名 (Renamed)
- C: 复制 (Copied)

## 错误处理
- 如果提供的哈希值无效，显示错误信息并退出
- 如果不在git仓库中，提示用户切换到正确的目录
- 如果哈希值不存在，显示相应的错误信息