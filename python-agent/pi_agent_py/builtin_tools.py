from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from .types import AgentTool, ToolContext, ToolResult


def _resolve(root: Path, raw: str) -> Path:
    root = root.resolve()
    path = (root / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
    if path != root and root not in path.parents:
        raise PermissionError(f'path escapes workspace: {raw}')
    return path


def coding_tools(workspace: str | Path = '.', *, bash_timeout: float = 30.0, max_output: int = 100_000) -> list[AgentTool]:
    root = Path(workspace).resolve()

    async def read_file(_: str, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = _resolve(root, args['path'])
        text = path.read_text(encoding='utf-8')
        start = max(1, int(args.get('start_line', 1)))
        end = int(args.get('end_line', 0))
        lines = text.splitlines()
        selected = lines[start - 1 : end or None]
        return ToolResult('\n'.join(selected), {'path': str(path), 'lines': len(selected)})

    async def write_file(_: str, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = _resolve(root, args['path'])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args['content'], encoding='utf-8')
        return ToolResult(f'wrote {path.relative_to(root)}', {'path': str(path), 'bytes': len(args['content'].encode())})

    async def edit_file(_: str, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = _resolve(root, args['path'])
        old = path.read_text(encoding='utf-8')
        needle = args['old_text']
        count = old.count(needle)
        if count == 0:
            raise ValueError('old_text not found')
        if count > 1 and not args.get('replace_all', False):
            raise ValueError(f'old_text occurs {count} times; set replace_all=true or provide more context')
        new = old.replace(needle, args['new_text']) if args.get('replace_all', False) else old.replace(needle, args['new_text'], 1)
        path.write_text(new, encoding='utf-8')
        return ToolResult(f'edited {path.relative_to(root)}', {'path': str(path), 'replacements': count if args.get('replace_all') else 1})

    async def bash(_: str, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        command = args['command']
        await ctx.update(f'running: {command}')
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=float(args.get('timeout', bash_timeout)))
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise TimeoutError(f'command timed out after {args.get("timeout", bash_timeout)}s')
        output = stdout.decode(errors='replace')
        if len(output) > max_output:
            output = output[:max_output] + '\n...[truncated]'
        if proc.returncode != 0:
            raise RuntimeError(f'exit code {proc.returncode}\n{output}')
        return ToolResult(output or '(no output)', {'returncode': proc.returncode})

    obj = {'type': 'object'}
    return [
        AgentTool(
            name='read', label='Read', description='Read a UTF-8 text file inside the workspace.',
            parameters={**obj, 'properties': {'path': {'type': 'string'}, 'start_line': {'type': 'integer'}, 'end_line': {'type': 'integer'}}, 'required': ['path']},
            execute=read_file,
        ),
        AgentTool(
            name='write', label='Write', description='Create or overwrite a UTF-8 text file inside the workspace.',
            parameters={**obj, 'properties': {'path': {'type': 'string'}, 'content': {'type': 'string'}}, 'required': ['path', 'content']},
            execute=write_file,
            execution_mode='sequential',
        ),
        AgentTool(
            name='edit', label='Edit', description='Replace exact text in a file inside the workspace.',
            parameters={**obj, 'properties': {'path': {'type': 'string'}, 'old_text': {'type': 'string'}, 'new_text': {'type': 'string'}, 'replace_all': {'type': 'boolean'}}, 'required': ['path', 'old_text', 'new_text']},
            execute=edit_file,
            execution_mode='sequential',
        ),
        AgentTool(
            name='bash', label='Bash', description='Run a shell command with the workspace as cwd.',
            parameters={**obj, 'properties': {'command': {'type': 'string'}, 'timeout': {'type': 'number'}}, 'required': ['command']},
            execute=bash,
            execution_mode='sequential',
        ),
    ]
