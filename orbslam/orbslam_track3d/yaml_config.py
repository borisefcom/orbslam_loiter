from __future__ import annotations

from pathlib import Path


def _parse_scalar(value: str) -> object:
    text = str(value).strip()
    if not text:
        return ""
    low = text.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "none", "~"):
        return None
    try:
        if "." in text or "e" in low:
            return float(text)
        return int(text)
    except Exception:
        return text.strip("\"'")


def load_yaml_dict(path: Path) -> dict[str, object]:
    """Load a YAML file into a dict.

    Uses PyYAML when available; falls back to a minimal indentation-based parser
    sufficient for our config files.
    """
    path = Path(path)
    if not path.exists():
        return {}

    try:
        import yaml  # type: ignore

        try:
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)  # type: ignore[attr-defined]
        except UnicodeDecodeError:
            with path.open("r", encoding="utf-8-sig") as f:
                data = yaml.safe_load(f)  # type: ignore[attr-defined]
        return data if isinstance(data, dict) else {}
    except Exception:
        pass

    data: dict[str, object] = {}
    stack: list[tuple[int, dict[str, object]]] = [(0, data)]
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return {}
    for raw in lines:
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, sep, rest = line.strip().partition(":")
        if not sep:
            continue
        while stack and indent < stack[-1][0]:
            stack.pop()
        if not stack:
            stack = [(0, data)]
        parent = stack[-1][1]
        if rest.strip():
            parent[key] = _parse_scalar(rest)
        else:
            child: dict[str, object] = {}
            parent[key] = child
            stack.append((indent + 1, child))
    return data

