"""
sys_stats.py

Lightweight system utilization snapshot for the RPi companion computer.

Designed for low overhead:
- Prefer procfs (/proc) instead of spawning `top` every sample.
- Returns a small dict suitable for UDP JSON telemetry.

If running on non-Linux platforms, returns {"available": False, ...}.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple


_last_cpu: Optional[Dict[str, Tuple[int, int]]] = None  # core -> (total_jiffies, idle_jiffies)
_last_cpu_t: float = 0.0


def _is_linux_procfs() -> bool:
    try:
        return os.name == "posix" and os.path.exists("/proc/stat") and os.path.exists("/proc/meminfo")
    except Exception:
        return False


def _read_proc_cpu_times() -> Dict[str, Tuple[int, int]]:
    """
    Return per-core totals from /proc/stat.
    Values are Linux jiffies since boot.
    """
    out: Dict[str, Tuple[int, int]] = {}
    with open("/proc/stat", "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("cpu"):
                continue
            # cpuN lines only (skip aggregated "cpu ")
            if len(line) < 4 or not line[3].isdigit():
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            name = str(parts[0])
            try:
                vals = [int(x) for x in parts[1:]]
            except Exception:
                continue
            total = int(sum(vals))
            idle = int(vals[3] + (vals[4] if len(vals) > 4 else 0))
            out[name] = (total, idle)
    return out


def _read_proc_meminfo_kb() -> Dict[str, int]:
    out: Dict[str, int] = {}
    with open("/proc/meminfo", "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if ":" not in line:
                continue
            k, rest = line.split(":", 1)
            k = k.strip()
            v0 = (rest.strip().split() or ["0"])[0]
            try:
                out[k] = int(v0)
            except Exception:
                continue
    return out


def _pct(used: float, total: float) -> float:
    if total <= 0.0:
        return 0.0
    return max(0.0, min(100.0, (used / total) * 100.0))


def system_snapshot(*, want_cores: int = 4) -> dict:
    """
    Best-effort system utilization snapshot.

    Returns:
      {
        "available": bool,
        "ok": bool,
        "source": "procfs",
        "cpu_pct": [core1..coreN],
        "mem_pct": float,
        "swap_pct": float,
        "err": Optional[str],
      }
    """
    if not _is_linux_procfs():
        return {"available": False, "ok": False, "source": "none", "cpu_pct": None, "mem_pct": None, "swap_pct": None, "err": "not_linux"}

    global _last_cpu, _last_cpu_t

    try:
        cur_cpu = _read_proc_cpu_times()
        mem = _read_proc_meminfo_kb()
    except Exception as e:
        return {"available": False, "ok": False, "source": "procfs", "cpu_pct": None, "mem_pct": None, "swap_pct": None, "err": f"read_error:{e!r}"}

    cpu_pct: List[float] = []
    try:
        prev = _last_cpu
        _last_cpu = cur_cpu
        _last_cpu_t = float(time.time())
        if prev:
            # cpu0..cpuN-1 -> core1..coreN
            for i in range(int(max(0, want_cores))):
                key = f"cpu{i}"
                if key not in cur_cpu or key not in prev:
                    cpu_pct.append(0.0)
                    continue
                t0, i0 = prev[key]
                t1, i1 = cur_cpu[key]
                dt = int(t1 - t0)
                di = int(i1 - i0)
                if dt <= 0:
                    cpu_pct.append(0.0)
                else:
                    busy = float(max(0, dt - di))
                    cpu_pct.append(_pct(busy, float(dt)))
        else:
            # First sample: no delta yet.
            cpu_pct = [0.0 for _ in range(int(max(0, want_cores)))]
    except Exception:
        cpu_pct = [0.0 for _ in range(int(max(0, want_cores)))]

    mem_total = float(mem.get("MemTotal", 0))
    mem_avail = float(mem.get("MemAvailable", 0))
    mem_used = max(0.0, mem_total - mem_avail)
    mem_pct = _pct(mem_used, mem_total)

    sw_total = float(mem.get("SwapTotal", 0))
    sw_free = float(mem.get("SwapFree", 0))
    sw_used = max(0.0, sw_total - sw_free)
    swap_pct = _pct(sw_used, sw_total) if sw_total > 0 else 0.0

    return {
        "available": True,
        "ok": True,
        "source": "procfs",
        "cpu_pct": cpu_pct,
        "mem_pct": float(mem_pct),
        "swap_pct": float(swap_pct),
        "err": None,
    }

