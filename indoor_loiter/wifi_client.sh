#!/usr/bin/env bash
set -euo pipefail

# Configure this Raspberry Pi as a Wi‑Fi client using NetworkManager.
#
# SSID: ESP-710 mesh
# PASS: ESP915mhz

IFACE="${IFACE:-wlan0}"
CLIENT_CONN="${CLIENT_CONN:-esp710_mesh}"
SSID="${SSID:-ESP-710 mesh}"
PASS="${PASS:-ESP915mhz}"
AP_CONN="${AP_CONN:-esp_drone_ap}"
PRIORITY="${PRIORITY:-500}"

if [[ "${EUID:-0}" -ne 0 ]]; then
  exec sudo -n --preserve-env=IFACE,CLIENT_CONN,SSID,PASS,AP_CONN "$0" "$@"
fi

if ! systemctl is-active --quiet NetworkManager; then
  echo "[wifi_client] NetworkManager is not active; cannot configure Wi‑Fi." >&2
  exit 1
fi

# Bring down the hotspot if it's up.
if nmcli -t -f NAME connection show --active | grep -Fxq "${AP_CONN}"; then
  nmcli -w 5 connection down "${AP_CONN}" || true
fi
if nmcli -t -f NAME connection show | grep -Fxq "${AP_CONN}"; then
  # Ensure AP doesn't win autoconnect on reboot when we're switching to client mode.
  nmcli connection modify "${AP_CONN}" connection.autoconnect-priority 0 || true
fi

if nmcli -t -f NAME connection show | grep -Fxq "${CLIENT_CONN}"; then
  nmcli connection modify "${CLIENT_CONN}" connection.autoconnect yes
  nmcli connection modify "${CLIENT_CONN}" connection.autoconnect-priority "${PRIORITY}" || true
  nmcli connection modify "${CLIENT_CONN}" connection.interface-name "${IFACE}" || true
  nmcli connection modify "${CLIENT_CONN}" 802-11-wireless.ssid "${SSID}"
  nmcli connection modify "${CLIENT_CONN}" wifi-sec.key-mgmt wpa-psk
  nmcli connection modify "${CLIENT_CONN}" wifi-sec.psk "${PASS}"
  nmcli connection modify "${CLIENT_CONN}" ipv4.method auto ipv6.method ignore
  nmcli -w 20 connection up "${CLIENT_CONN}"
else
  # This creates the profile and connects.
  nmcli -w 20 dev wifi connect "${SSID}" password "${PASS}" ifname "${IFACE}" name "${CLIENT_CONN}"
  nmcli connection modify "${CLIENT_CONN}" connection.autoconnect yes
  nmcli connection modify "${CLIENT_CONN}" connection.autoconnect-priority "${PRIORITY}" || true
fi

echo "[wifi_client] Connected: SSID=\"${SSID}\" (conn=\"${CLIENT_CONN}\" iface=\"${IFACE}\")"
ip -brief addr show "${IFACE}" || true
nmcli -f GENERAL.STATE,GENERAL.CONNECTION dev show "${IFACE}" || true
