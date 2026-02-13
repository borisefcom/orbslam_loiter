#!/usr/bin/env python3
# cellular.py — cellular snapshot using mmcli (human output) + nmcli
# - Provides cellular_snapshot() for telemetry
# - When run directly, prints a single JSON snapshot (no CLI)

import json
import re
import shutil
import subprocess
import time
from typing import Dict, Any, List, Optional, Tuple

MMCLI = "mmcli"
NMCLI = "nmcli"

# -------------------- small helpers --------------------

def _run_cmd(args: List[str], timeout: float = 3.0) -> Tuple[int, str, str]:
    try:
        cp = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return cp.returncode, (cp.stdout or "").strip(), (cp.stderr or "").strip()
    except Exception as e:
        return 1, "", str(e)

def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None

def _now_ms() -> int:
    return int(time.time() * 1000)

def _to_int(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    try:
        return int(s)
    except Exception:
        return None

# Strip ANSI color/escape sequences (e.g., \x1b[32mconnected\x1b[0m)
_ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

def _deansi(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    return _ANSI_RE.sub("", s)

# -------------------- mmcli (human) parsing --------------------

_h_re = re.compile(r"^\s*([A-Za-z0-9 /-]+)\s*:\s*(.*)$")

def _mmcli_modem_human(modem_index: int) -> Optional[str]:
    rc, out, _ = _run_cmd([MMCLI, "-m", str(modem_index)])
    if rc != 0 or not out:
        return None
    return out

def _mm_h_extract_sections(text: str) -> Dict[str, Dict[str, str]]:
    """
    Parse the human mmcli -m output into sections: 'Status', 'System', '3GPP', ...
    Each section is a dict of lowercased key -> value (ANSI stripped).
    """
    sections: Dict[str, Dict[str, str]] = {}
    current: Optional[str] = None
    for line in text.splitlines():
        line = line.rstrip()
        if not line:
            continue
        if set(line.strip()) == set("-"):  # separator line (-----)
            continue
        if "|" in line:
            left, right = line.split("|", 1)
            section = left.strip()
            kv = right.strip()
            if section:
                current = section
                sections.setdefault(current, {})
            if current and kv:
                m = _h_re.match(kv)
                if m:
                    key = m.group(1).strip().lower()
                    val = _deansi(m.group(2).strip())  # strip ANSI here
                    sections[current][key] = val
    return sections

def _parse_status(sec_status: Dict[str, str]) -> Dict[str, Any]:
    # state: connected | ... ; power state: on; access tech: lte; signal quality: "19% (cached)"
    state_raw = sec_status.get("state")
    state = _deansi(state_raw)
    power = _deansi(sec_status.get("power state"))
    access = _deansi(sec_status.get("access tech"))
    sigq_raw = _deansi(sec_status.get("signal quality"))
    sigq = None
    if sigq_raw:
        m = re.search(r"(\d+)\s*%", sigq_raw)
        if m:
            sigq = _to_int(m.group(1))
    return {
        "state": state if state else None,
        "power": power.lower() if power else None,
        "access_tech": access.lower() if access else None,
        "signal_quality": sigq
    }

def _parse_3gpp(sec_3gpp: Dict[str, str]) -> Dict[str, Any]:
    # operator id, operator name, registration
    return {
        "state": _deansi(sec_3gpp.get("registration")) or None,
        "operator_id": _deansi(sec_3gpp.get("operator id")) or None,
        "operator_name": _deansi(sec_3gpp.get("operator name")) or None,
    }

def _parse_iface_from_system(sec_system: Dict[str, str]) -> Optional[str]:
    # Find token like "wwan0 (net)" inside "ports"
    ports = sec_system.get("ports")
    if ports:
        tokens = [t.strip() for t in ports.split(",")]
        for t in tokens:
            m = re.match(r"([a-zA-Z0-9_.-]+)\s*\(net\)", t)
            if m:
                return m.group(1)
    return None

# -------------------- nmcli helpers --------------------

def _nmcli_iface_ipv4(iface: str) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
    """
    Returns (ipv4_dict, mtu) for iface using nmcli.
    ipv4_dict = {"address": "...", "prefix": int, "gateway": "...", "dns": [...]}
    """
    if not _have(NMCLI) or not iface:
        return None, None

    # IPv4 ADDRESS (may have multiple lines; pick first)
    rc_a, out_a, _ = _run_cmd([NMCLI, "-g", "IP4.ADDRESS", "device", "show", iface])
    address = None
    prefix = None
    if rc_a == 0 and out_a:
        for line in out_a.splitlines():
            line = line.strip()
            if not line:
                continue
            if "/" in line:
                address, p = line.split("/", 1)
                address = address.strip()
                prefix = _to_int(p.strip())
                break
            else:
                address = line.strip()
                break

    # Gateway
    rc_g, out_g, _ = _run_cmd([NMCLI, "-g", "IP4.GATEWAY", "device", "show", iface])
    gateway = out_g.splitlines()[0].strip() if (rc_g == 0 and out_g.strip()) else None

    # DNS (multiple lines)
    rc_d, out_d, _ = _run_cmd([NMCLI, "-g", "IP4.DNS", "device", "show", iface])
    dns_list: List[str] = []
    if rc_d == 0 and out_d:
        for line in out_d.splitlines():
            s = line.strip()
            if s:
                dns_list.append(s)

    # MTU
    rc_m, out_m, _ = _run_cmd([NMCLI, "-g", "GENERAL.MTU", "device", "show", iface])
    mtu = _to_int(out_m.splitlines()[0].strip()) if (rc_m == 0 and out_m.strip()) else None

    ipv4 = None
    if address:
        ipv4 = {"address": address, "prefix": prefix, "gateway": gateway, "dns": dns_list or None}

    return ipv4, mtu

# -------------------- public API --------------------

def cellular_snapshot(modem_index: int = 0) -> Dict[str, Any]:
    """
    Normalized snapshot matching your schema:
      type, ts, available, status{state,power,access_tech,signal_quality},
      signal{dbm?}, registration{state,operator_id,operator_name},
      connection{connected,iface,ipv4,ipv6:null,mtu}
    """
    if not _have(MMCLI):
        return {
            "type": "cellular",
            "ts": _now_ms(),
            "available": False,
            "status": {"state": None, "power": None, "access_tech": None, "signal_quality": None},
            "signal": {"dbm": None},
            "registration": {"state": None, "operator_id": None, "operator_name": None},
            "connection": {"connected": False, "iface": None, "ipv4": None, "ipv6": None, "mtu": None},
            "reason": "mmcli not found",
        }

    human = _mmcli_modem_human(modem_index)
    if not human:
        return {
            "type": "cellular",
            "ts": _now_ms(),
            "available": False,
            "status": {"state": None, "power": None, "access_tech": None, "signal_quality": None},
            "signal": {"dbm": None},
            "registration": {"state": None, "operator_id": None, "operator_name": None},
            "connection": {"connected": False, "iface": None, "ipv4": None, "ipv6": None, "mtu": None},
        }

    secs = _mm_h_extract_sections(human)
    status = _parse_status(secs.get("Status", {}))
    reg = _parse_3gpp(secs.get("3GPP", {}))
    iface = _parse_iface_from_system(secs.get("System", {}))

    # Determine connected flag robustly (after de-ANSI + lowercase)
    state_norm = (status.get("state") or "").strip().lower()
    connected = ("connected" in state_norm)

    # dBm is best-effort (some stacks need --signal-setup)
    dbm = None
    rc_s, out_s, _ = _run_cmd([MMCLI, "-m", str(modem_index), "--signal-get", "--output-keyvalue"])
    if rc_s == 0 and out_s:
        for line in out_s.splitlines():
            if "signal.dbm=" in line:
                try:
                    dbm = _to_int(line.split("=", 1)[1].strip())
                    break
                except Exception:
                    pass

    ipv4 = None
    mtu = None
    if connected:
        if not iface:
            # heuristic fallback commonly valid for MBIM/QMI:
            iface = "wwan0"
        ipv4, mtu = _nmcli_iface_ipv4(iface)

    return {
        "type": "cellular",
        "ts": _now_ms(),
        "available": True,
        "status": status,
        "signal": {"dbm": dbm},
        "registration": reg,
        "connection": {
            "connected": bool(connected),
            "iface": iface,
            "ipv4": ipv4,
            "ipv6": None,
            "mtu": mtu,
        },
    }

# -------------------- print once when run --------------------

if __name__ == "__main__":
    print(json.dumps(cellular_snapshot(0), separators=(",", ":")))
