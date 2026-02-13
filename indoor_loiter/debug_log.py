"""
Simple in-memory debug logger shared across modules.
"""

from typing import List
import threading

_buf: List[str] = []
_lock = threading.Lock()


def log(msg: str) -> None:
    """Append debug line to shared buffer."""
    try:
        with _lock:
            _buf.append(str(msg))
    except Exception:
        pass


def clear() -> None:
    """Clear the buffer."""
    try:
        with _lock:
            _buf.clear()
    except Exception:
        pass


def flush(path: str = "result.txt", append: bool = False) -> None:
    """
    Flush buffer to a file and clear it.
    If append is False (default), overwrite the file with the latest buffer.
    """
    try:
        with _lock:
            if not _buf:
                return
            lines = list(_buf)
            _buf.clear()
        mode = "a" if append else "w"
        with open(path, mode, encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
    except Exception:
        pass
