"""Shared helpers for multiprocessing pipeline stages."""

from __future__ import annotations

def _proc_setup_signals(*, stop_event) -> None:
    try:
        import signal

        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            signal.signal(signal.SIGTERM, lambda *_a: stop_event.set())  # type: ignore[misc]
        except Exception:
            pass
    except Exception:
        pass


