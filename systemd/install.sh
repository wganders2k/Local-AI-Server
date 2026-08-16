#!/usr/bin/env bash
#
# Make the compose stack survive reboots across NVIDIA driver upgrades.
#
# Fixes three things:
#
#   1. Installs local-ai-server.service, which recreates containers at boot if a
#      plain start fails. This is what actually rescues the stack after a driver
#      upgrade invalidates the baked-in library mount paths.
#
#   2. Orders that unit after nvidia-cdi-refresh.service, closing a boot-time
#      race where GPU containers could start before the toolkit finished
#      regenerating its CDI spec for the current driver and fell back to a
#      stale one. Found 2026-08-16 after this silently held the stack down
#      across three separate reboots — see the comment in the unit file.
#
#   3. Stops unattended-upgrades from swapping the NVIDIA driver out from under
#      running containers in the first place. Driver upgrades on a GPU box
#      should be deliberate: they break every running GPU container until it is
#      recreated, and can desync the kernel module from userspace until reboot.
#
# Run:  sudo bash systemd/install.sh
#
set -euo pipefail

UNIT_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/local-ai-server.service"
UNIT_DST=/etc/systemd/system/local-ai-server.service
UU_CONF=/etc/apt/apt.conf.d/52-nvidia-hold

if [[ $EUID -ne 0 ]]; then
    echo "Run with sudo: sudo bash $0" >&2
    exit 1
fi

# -- 1. boot unit --

install -m 0644 "$UNIT_SRC" "$UNIT_DST"
systemctl daemon-reload
systemctl enable local-ai-server.service
echo "Installed  -> $UNIT_DST (enabled)"

# -- 2. keep unattended-upgrades away from the driver --

if [[ -f "$UU_CONF" ]]; then
    echo "Exists     -> $UU_CONF (left as-is)"
else
    cat > "$UU_CONF" <<'EOF'
// Keep NVIDIA driver/CUDA packages out of unattended upgrades.
//
// Upgrading the driver replaces the versioned userspace libraries that running
// GPU containers have bind-mounted by exact path. Every such container then
// fails to start until it is recreated, and the kernel module can be out of
// sync with userspace until the next reboot. On a box whose whole job is GPU
// inference, that upgrade should be a decision, not a surprise at 6am.
//
// Upgrade deliberately instead:
//     sudo apt update && sudo apt install --only-upgrade 'nvidia-driver-*'
//     sudo reboot
//     cd /home/peacow/local-ai-server && docker compose up -d --force-recreate
Unattended-Upgrade::Package-Blacklist {
    "nvidia";
    "libnvidia";
    "cuda";
};
EOF
    echo "Wrote      -> $UU_CONF"
fi

if ! apt-config dump >/dev/null 2>&1; then
    echo "WARNING: apt-config could not parse the configuration — check $UU_CONF" >&2
    exit 1
fi
echo "Validated  -> apt config parses"

echo
echo "Blacklist now in effect:"
apt-config dump 2>/dev/null | grep -i "Package-Blacklist::" | sed 's/^/    /' || true

echo
echo "Verify the boot unit without rebooting:"
echo "    sudo systemctl start local-ai-server.service"
echo "    systemctl status local-ai-server.service"
