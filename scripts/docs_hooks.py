"""MkDocs build hook: rewrite docs links that escape the docs directory.

`docs/` 中的模型与迁移文档大量相对链接指向仓库根的 configs/、src/、tests/、
ROADMAP.md 等文件。这些目标不在 docs_dir 内,MkDocs 不会复制它们,站点上会
404。本 hook 在渲染前把此类链接改写为 GitHub 绝对 URL;仓库内源文档保持不变。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from urllib.parse import quote

# src/detrs 的 Google 风格 docstring 大量沿用 Paddle 时期的无标注参数写法,
# griffe 的 "No type or annotation" 提示不影响站点渲染,降级以保持 --strict 可用。
# mkdocstrings 会把 griffe 的 logger 重定位到 "mkdocs.plugins.griffe"。
logging.getLogger("mkdocs.plugins.griffe").setLevel(logging.ERROR)

_BRANCH = "main"
_LINK = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")


def _github_url(repo_url: str, repo_root: Path, target: Path, is_dir: bool) -> str:
    kind = "tree" if is_dir else "blob"
    rel = quote(target.resolve().relative_to(repo_root).as_posix())
    return f"{repo_url.rstrip('/')}/{kind}/{_BRANCH}/{rel}"


def _rewrite(
    markdown: str, *, src_path: Path, repo_root: Path, docs_dir: Path, repo_url: str
) -> str:
    lines = markdown.splitlines(keepends=True)
    output = []
    fence: tuple[str, int] | None = None
    for line in lines:
        marker = _FENCE.match(line)
        if fence is not None:
            if (
                marker
                and marker.group(1)[0] == fence[0]
                and len(marker.group(1)) >= fence[1]
            ):
                fence = None
            output.append(line)
            continue
        if marker:
            fence = (marker.group(1)[0], len(marker.group(1)))
            output.append(line)
            continue

        def replace(match: re.Match) -> str:
            raw = match.group(1).strip()
            if "://" in raw or raw.startswith(("mailto:", "/", "#")):
                return match.group(0)
            path_part, _, fragment = raw.partition("#")
            resolved = (src_path.parent / path_part).resolve()
            if resolved == docs_dir or docs_dir in resolved.parents:
                return match.group(0)
            if not resolved.exists():
                return match.group(0)
            url = _github_url(repo_url, repo_root, resolved, path_part.endswith("/"))
            suffix = f"#{fragment}" if fragment else ""
            return match.group(0).replace(f"({match.group(1)})", f"({url}{suffix})")

        output.append(_LINK.sub(replace, line))
    return "".join(output)


def on_page_markdown(markdown: str, *, page, config, **kwargs) -> str:
    docs_dir = Path(config["docs_dir"]).resolve()
    repo_root = Path(config["config_file_path"]).resolve().parent
    return _rewrite(
        markdown,
        src_path=Path(page.file.abs_src_path),
        repo_root=repo_root,
        docs_dir=docs_dir,
        repo_url=config["repo_url"],
    )
