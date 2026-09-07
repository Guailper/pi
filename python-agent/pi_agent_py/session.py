from __future__ import annotations

import json
from pathlib import Path

from .types import Message


def save_session(path: str | Path, messages: list[Message]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps([m.to_dict() for m in messages], ensure_ascii=False, indent=2), encoding='utf-8')


def load_session(path: str | Path) -> list[Message]:
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    return [Message.from_dict(item) for item in data]
