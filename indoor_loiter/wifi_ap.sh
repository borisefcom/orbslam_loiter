#!/usr/bin/env bash
set -euo pipefail

# Configure this Raspberry Pi as a Wi‑Fi hotspot using NetworkManager.
#
# SSID: esp_drone
# PASS: ESP915mhz (note: WPA2 requires >= 8 chars; script will pad if shorter)
#
# Note: NetworkManager "shared" mode usually assigns the AP interface an IP like
# 10.42.0.1/24 and runs a DHCP server for connected clients.

IFACE="${IFACE:-wlan0}"
AP_CONN="${AP_CONN:-esp_drone_ap}"
SSID="${SSID:-esp_drone}"
PASS="${PASS:-ESP915mhz}"
PRIORITY="${PRIORITY:-500}"

if [[ "${EUID:-0}" -ne 0 ]]; then
  exec sudo -n --preserve-env=IFACE,AP_CONN,SSID,PASS "$0" "$@"
fi

if ! systemctl is-active --quiet NetworkManager; then
  echo "[wifi_ap] NetworkManager is not active; cannot configure hotspot." >&2
  exit 1
fi

# WPA2-PSK requires 8..63 ASCII chars. "esp710" is too short (6),
# so pad with zeros to keep the requested prefix while staying valid.
if [[ "${#PASS}" -lt 8 ]]; then
  orig="${PASS}"
  while [[ "${#PASS}" -lt 8 ]]; do
    PASS="${PASS}0"
  done
  echo "[wifi_ap] NOTE: WPA2-PSK requires 8..63 chars; '${orig}' padded -> '${PASS}'" >&2
fi

# Disconnect any active Wi‑Fi client connection on this interface.
# (This may drop SSH if you're connected over Wi‑Fi.)
active_wifi="$(nmcli -t -f NAME,TYPE,DEVICE connection show --active | awk -F: -v IFACE="${IFACE}" '$2=="802-11-wireless" && $3==IFACE {print $1}')"
if [[ -n "${active_wifi}" ]]; then
  while IFS= read -r c; do
    [[ -z "${c}" ]] && continue
    [[ "${c}" == "${AP_CONN}" ]] && continue
    nmcli -w 5 connection down "${c}" || true
  done <<< "${active_wifi}"
fi

if nmcli -t -f NAME connection show | grep -Fxq "${AP_CONN}"; then
  nmcli connection modify "${AP_CONN}" connection.autoconnect yes
  nmcli connection modify "${AP_CONN}" connection.autoconnect-priority "${PRIORITY}" || true
  nmcli connection modify "${AP_CONN}" 802-11-wireless.ssid "${SSID}"
  nmcli connection modify "${AP_CONN}" 802-11-wireless.mode ap
  nmcli connection modify "${AP_CONN}" 802-11-wireless.band bg
  nmcli connection modify "${AP_CONN}" 802-11-wireless.channel 6
  nmcli connection modify "${AP_CONN}" wifi-sec.key-mgmt wpa-psk
  nmcli connection modify "${AP_CONN}" wifi-sec.psk "${PASS}"
  nmcli connection modify "${AP_CONN}" ipv4.method shared ipv6.method ignore
else
  nmcli connection add type wifi ifname "${IFACE}" con-name "${AP_CONN}" autoconnect yes ssid "${SSID}"
  nmcli connection modify "${AP_CONN}" connection.autoconnect-priority "${PRIORITY}" || true
  nmcli connection modify "${AP_CONN}" 802-11-wireless.mode ap
  nmcli connection modify "${AP_CONN}" 802-11-wireless.band bg
  nmcli connection modify "${AP_CONN}" 802-11-wireless.channel 6
  nmcli connection modify "${AP_CONN}" ipv4.method shared ipv6.method ignore
  nmcli connection modify "${AP_CONN}" wifi-sec.key-mgmt wpa-psk
  nmcli connection modify "${AP_CONN}" wifi-sec.psk "${PASS}"
fi

nmcli -w 20 connection up "${AP_CONN}"

echo "[wifi_ap] Hotspot up: SSID=\"${SSID}\" (conn=\"${AP_CONN}\" iface=\"${IFACE}\")"
echo "[wifi_ap] Hotspot password: \"${PASS}\""
ip -brief addr show "${IFACE}" || true
nmcli -f GENERAL.STATE,GENERAL.CONNECTION dev show "${IFACE}" || true
echo "[wifi_ap] Default hotspot IP is usually 10.42.0.1 (NetworkManager shared mode)."
