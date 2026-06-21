#!/usr/bin/env bash
# Create stable /dev symlinks for the SO-101 arms and cameras via udev rules.
#
# USB device numbering (/dev/ttyACMx, /dev/videoN) is assigned at boot order and
# changes across reboots/replugs. This pins each device to a fixed name by its
# USB vendor/product/serial, so the rollout/record/teleop scripts never break:
#
#   /dev/lerobot_follower   follower arm   (CH340 serial 5A7A018247)
#   /dev/lerobot_leader     leader arm     (CH340 serial 5A7A017687)
#   /dev/lerobot_overhead   C920 webcam    (serial 724DAD5F)
#   /dev/lerobot_gripper    USB 2.0 camera (05a3:9230, no serial -> matched by model)
#
# Usage:  sudo ./setup_udev.sh
# Re-run any time hardware changes; it overwrites the rules file.
#
# To re-map after swapping hardware, edit the match values below. Find a device's
# attributes with:  udevadm info -q property -n /dev/ttyACM0   (or -n /dev/videoN)

set -euo pipefail

RULES_FILE="/etc/udev/rules.d/99-lerobot.rules"

# ── Device match table (verified 2026-06-21) ──────────────────────────────────
# Arms: same vendor/product (1a86:55d3), distinguished by serial.
FOLLOWER_SERIAL="5A7A018247"
LEADER_SERIAL="5A7A017687"
ARM_VENDOR="1a86"
ARM_PRODUCT="55d3"

# Overhead: Logitech C920, has a unique serial.
OVERHEAD_VENDOR="046d"
OVERHEAD_PRODUCT="08e5"
OVERHEAD_SERIAL="724DAD5F"

# Gripper: generic "USB 2.0 Camera" exposes no serial, so it is matched by
# vendor/product only. This is unambiguous as long as exactly one camera of this
# model is connected. If you add a second identical camera, pin it by USB port
# path (KERNELS=="...") instead.
GRIPPER_VENDOR="05a3"
GRIPPER_PRODUCT="9230"

# ── Require root ──────────────────────────────────────────────────────────────

if [[ "${EUID}" -ne 0 ]]; then
  echo "This script writes to ${RULES_FILE} and needs root. Re-running with sudo..."
  exec sudo -- "$0" "$@"
fi

# ── Write rules ───────────────────────────────────────────────────────────────
# ATTR{index}=="0" selects each camera's capture node (cameras also expose an
# index=1 metadata node we must not symlink). MODE="0666" lets lerobot open the
# device without sudo.

echo "Writing ${RULES_FILE}..."
cat > "${RULES_FILE}" <<EOF
# Managed by setup_udev.sh — stable names for SO-101 arms and cameras.

# ── Arms (SO-101, CH340 ${ARM_VENDOR}:${ARM_PRODUCT}) ──
SUBSYSTEM=="tty", ATTRS{idVendor}=="${ARM_VENDOR}", ATTRS{idProduct}=="${ARM_PRODUCT}", ATTRS{serial}=="${FOLLOWER_SERIAL}", SYMLINK+="lerobot_follower", MODE="0666"
SUBSYSTEM=="tty", ATTRS{idVendor}=="${ARM_VENDOR}", ATTRS{idProduct}=="${ARM_PRODUCT}", ATTRS{serial}=="${LEADER_SERIAL}", SYMLINK+="lerobot_leader", MODE="0666"

# ── Overhead camera (Logitech C920 ${OVERHEAD_VENDOR}:${OVERHEAD_PRODUCT}) ──
SUBSYSTEM=="video4linux", ATTRS{idVendor}=="${OVERHEAD_VENDOR}", ATTRS{idProduct}=="${OVERHEAD_PRODUCT}", ATTRS{serial}=="${OVERHEAD_SERIAL}", ATTR{index}=="0", SYMLINK+="lerobot_overhead", MODE="0666"

# ── Gripper camera (USB 2.0 Camera ${GRIPPER_VENDOR}:${GRIPPER_PRODUCT}, no serial) ──
SUBSYSTEM=="video4linux", ATTRS{idVendor}=="${GRIPPER_VENDOR}", ATTRS{idProduct}=="${GRIPPER_PRODUCT}", ATTR{index}=="0", SYMLINK+="lerobot_gripper", MODE="0666"
EOF

# ── Reload and apply ──────────────────────────────────────────────────────────

echo "Reloading udev rules..."
udevadm control --reload-rules
udevadm trigger --subsystem-match=tty --subsystem-match=video4linux

sleep 1  # give udev a moment to create the symlinks

# ── Verify ────────────────────────────────────────────────────────────────────

echo
echo "Resulting symlinks:"
ok=true
for name in lerobot_follower lerobot_leader lerobot_overhead lerobot_gripper; do
  if [[ -e "/dev/${name}" ]]; then
    printf "  /dev/%-18s -> %s\n" "${name}" "$(readlink -f "/dev/${name}")"
  else
    printf "  /dev/%-18s MISSING (device not connected, or match values need updating)\n" "${name}"
    ok=false
  fi
done

echo
if ${ok}; then
  echo "All devices mapped. Scripts can now use the stable /dev/lerobot_* names."
else
  echo "Some symlinks are missing. Plug in the device(s) and re-run, or check the"
  echo "match values with: udevadm info -q property -n /dev/ttyACMx (or /dev/videoN)"
fi
