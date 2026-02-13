from __future__ import annotations

import json
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional


def _json_dumps_ascii(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"), sort_keys=False)


@dataclass
class HeadlessIoSettings:
    enabled: bool = False
    transport: str = "stdio_jsonl"  # stdio_jsonl|none
    stdin_enabled: bool = True
    stdout_enabled: bool = True
    max_in_per_tick: int = 50
    eof_is_shutdown: bool = True


class HeadlessJsonIo:
    """
    Minimal JSONL stdin/stdout shim compatible with Drone_client's headless protocol.

    Input (stdin):
      - One JSON dict per line, with at least {"cmd": "..."}.

    Output (stdout):
      - One JSON dict per line:
          {"type":"<event>","ts_wall":<unix_seconds>,"payload":{...}}

    Notes:
      - Uses `sys.__stdout__` for output so JSON events can bypass any print() redirection.
      - Encoding is ASCII-only (`ensure_ascii=True`).
    """

    def __init__(self, *, settings: HeadlessIoSettings) -> None:
        self.settings = settings
        self._in_q: "queue.Queue[dict]" = queue.Queue(maxsize=512)
        self._alive = threading.Event()
        self._stdin_thread: Optional[threading.Thread] = None

        if not bool(self.settings.enabled):
            return
        if str(self.settings.transport).strip().lower() != "stdio_jsonl":
            return

        if bool(self.settings.stdin_enabled):
            self._alive.set()
            th = threading.Thread(target=self._stdin_reader_main, name="stdin_jsonl", daemon=True)
            th.start()
            self._stdin_thread = th

    def close(self) -> None:
        try:
            self._alive.clear()
        except Exception:
            pass

    def poll_in(self) -> list[dict]:
        out: list[dict] = []
        if not bool(self.settings.enabled):
            return out
        n = int(max(0, int(self.settings.max_in_per_tick)))
        for _ in range(n):
            try:
                msg = self._in_q.get_nowait()
            except Exception:
                break
            if isinstance(msg, dict):
                out.append(dict(msg))
        return out

    def send(self, msg: dict) -> None:
        if not bool(self.settings.enabled):
            return
        if not bool(self.settings.stdout_enabled):
            return
        try:
            line = _json_dumps_ascii(dict(msg))
        except Exception:
            return
        try:
            sys.__stdout__.write(str(line) + "\n")
            sys.__stdout__.flush()
        except Exception:
            try:
                sys.stdout.write(str(line) + "\n")
                sys.stdout.flush()
            except Exception:
                pass

    def send_event(self, *, event: str, payload: dict) -> None:
        try:
            self.send({"type": str(event), "ts_wall": float(time.time()), "payload": dict(payload)})
        except Exception:
            return

    def _stdin_reader_main(self) -> None:
        try:
            st = sys.stdin
        except Exception:
            st = None
        if st is None:
            return

        while self._alive.is_set():
            try:
                line = st.readline()
            except Exception:
                break
            if not line:
                if bool(self.settings.eof_is_shutdown):
                    try:
                        self._in_q.put_nowait({"cmd": "shutdown", "reason": "stdin_eof"})
                    except Exception:
                        pass
                break
            s = str(line).strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            try:
                self._in_q.put_nowait(dict(obj))
            except Exception:
                try:
                    _ = self._in_q.get_nowait()
                except Exception:
                    pass
                try:
                    self._in_q.put_nowait(dict(obj))
                except Exception:
                    pass

