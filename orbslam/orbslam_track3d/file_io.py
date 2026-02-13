from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def write_text_atomic(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(str(text), encoding=str(encoding))
    os.replace(str(tmp), str(path))


def write_json_atomic(path: Path, data: Any) -> None:
    write_text_atomic(
        Path(path),
        json.dumps(data, ensure_ascii=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

