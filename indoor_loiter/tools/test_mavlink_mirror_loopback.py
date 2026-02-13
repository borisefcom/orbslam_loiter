#!/usr/bin/env python3
from __future__ import annotations

"""
Sanity-check `pymavlink` + `MavlinkBridge` mirror path without a real FCU.

This spins up a local (127.0.0.1) MAVLink "FCU input" UDP port using
`px4.serial = udpin:<host>:<port_in>`, enables `px4.mirror` to a second UDP
port, sends a few HEARTBEAT messages into the input port, and verifies the
mirror receives at least one MAVLink packet.

Usage:
  python3 tools/test_mavlink_mirror_loopback.py
"""

import argparse
import socket
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pymavlink import mavutil  # type: ignore

from mavlink_bridge import MavlinkBridge


def _pick_free_udp_port(host: str) -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind((host, 0))
        return int(s.getsockname()[1])
    finally:
        try:
            s.close()
        except Exception:
            pass


def _recv_one(conn, timeout_s: float) -> Optional[object]:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        msg = conn.recv_match(blocking=True, timeout=0.25)
        if msg is None:
            continue
        try:
            if msg.get_type() == "BAD_DATA":
                continue
        except Exception:
            pass
        return msg
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--in-port", type=int, default=0, help="FCU input UDP port (0=auto)")
    ap.add_argument("--mirror-port", type=int, default=0, help="Mirror UDP port (0=auto)")
    ap.add_argument("--count", type=int, default=8, help="Number of heartbeats to send")
    ap.add_argument("--timeout-s", type=float, default=4.0, help="Mirror receive timeout")
    args = ap.parse_args()

    host = str(args.host).strip() or "127.0.0.1"
    in_port = int(args.in_port) if int(args.in_port) > 0 else _pick_free_udp_port(host)
    mirror_port = int(args.mirror_port) if int(args.mirror_port) > 0 else _pick_free_udp_port(host)
    count = int(max(1, int(args.count)))
    timeout_s = float(max(0.25, float(args.timeout_s)))

    cfg = {
        "px4": {
            "enabled": True,
            "serial": f"udpin:{host}:{in_port}",
            "baud": 115200,
            "dialect": "common",
            "mavlink2": True,
            "mirror": {"enabled": True, "udp": f"{host}:{mirror_port}"},
        }
    }

    mb = MavlinkBridge(cfg, print_fn=lambda *a, **k: None)
    mb.start()

    rx = None
    tx = None
    try:
        rx = mavutil.mavlink_connection(f"udpin:{host}:{mirror_port}")
        tx = mavutil.mavlink_connection(f"udpout:{host}:{in_port}")

        # MavlinkBridge's RX loop does an initial `wait_heartbeat()` which consumes
        # the first heartbeat. Send a few to ensure at least one is mirrored.
        for _ in range(count):
            tx.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                mavutil.mavlink.MAV_AUTOPILOT_GENERIC,
                0,
                0,
                0,
            )
            time.sleep(0.15)

        msg = _recv_one(rx, timeout_s=timeout_s)
        if msg is None:
            print(
                "FAIL: no mirrored MAVLink packet received",
                {"host": host, "in_port": in_port, "mirror_port": mirror_port},
                flush=True,
            )
            return 2

        try:
            mtype = str(msg.get_type())
        except Exception:
            mtype = "UNKNOWN"
        print(
            "OK: pymavlink + mirror functional",
            {"host": host, "in_port": in_port, "mirror_port": mirror_port, "mirrored_type": mtype},
            flush=True,
        )
        return 0
    finally:
        try:
            if tx is not None:
                tx.close()
        except Exception:
            pass
        try:
            if rx is not None:
                rx.close()
        except Exception:
            pass
        try:
            mb.stop()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
