#!/usr/bin/env bash
# install-arch.sh — install CW Scanner's dependencies on Arch Linux.
#
# What it does:
#   1. Installs rtl-sdr, numpy and scipy from the official repos (pacman).
#   2. Creates a virtualenv in .venv that can see those system packages, and
#      pip-installs pyrtlsdr into it (pyrtlsdr isn't in the official repos,
#      and Arch's system Python refuses plain `pip install`).
#   3. Checks that the DVB-T kernel driver isn't holding the dongle.
#   4. Runs the synthetic self-test.
#
# Safe to run again: pacman skips what's installed, the venv is reused.
#
# Usage:
#   ./install-arch.sh
#   source .venv/bin/activate
#   python cwscan.py scan

set -euo pipefail

cd "$(dirname "$0")"
VENV=.venv
# Use Arch's own Python explicitly: pacman's numpy/scipy are built for it,
# and a `python` from mise/pyenv/asdf earlier in PATH may be a different
# version that can't see them.
SYS_PYTHON=/usr/bin/python

if ! command -v pacman >/dev/null; then
    echo "pacman not found — this script is for Arch Linux." >&2
    exit 1
fi

echo "==> Installing system packages (needs sudo)"
sudo pacman -S --needed --noconfirm rtl-sdr python python-numpy python-scipy

echo "==> Setting up virtualenv in $VENV"
# --system-site-packages lets the venv use pacman's numpy/scipy instead of
# downloading and building its own copies.
# A venv left over from a different Python is rebuilt.
if [ -d "$VENV" ] && ! grep -qx "home = /usr/bin" "$VENV/pyvenv.cfg"; then
    echo "  Existing $VENV wasn't made with $SYS_PYTHON — recreating it."
    rm -rf "$VENV"
fi
if [ ! -d "$VENV" ]; then
    "$SYS_PYTHON" -m venv --system-site-packages "$VENV"
fi
"$VENV/bin/python" -m pip install --quiet --upgrade pip
# pyrtlsdr is pinned to 0.3.0: 0.4.0 and later call rtlsdr_set_dithering,
# which only exists in the RTL-SDR Blog fork of librtlsdr, not in Arch's
# upstream (osmocom) rtl-sdr package. 0.3.0 still imports pkg_resources,
# which setuptools 81 removed, hence the setuptools pin.
"$VENV/bin/python" -m pip install --quiet "pyrtlsdr==0.3.0" "setuptools<81"

echo "==> Checking the dongle setup"
# Arch's rtl-sdr package blacklists the DVB-T driver, but the blacklist only
# takes effect for modules loaded after it was installed.
if lsmod | grep -q '^dvb_usb_rtl28xxu'; then
    echo "  The DVB-T driver (dvb_usb_rtl28xxu) is loaded and will block the"
    echo "  dongle. Unload it with:  sudo modprobe -r dvb_usb_rtl28xxu"
    echo "  (or just unplug the dongle and plug it back in, or reboot)."
fi
# USB permissions come from rtl-sdr's udev rules (uaccess tag), so the
# logged-in user gets access with no group changes needed.
if command -v rtl_test >/dev/null && timeout 3 rtl_test -t >/dev/null 2>&1; then
    echo "  RTL-SDR dongle found."
else
    echo "  No RTL-SDR dongle detected (fine if it isn't plugged in yet)."
fi

echo "==> Running self-test"
"$VENV/bin/python" test_synth.py

echo
echo "Done. To use the scanner:"
echo "  source $VENV/bin/activate"
echo "  python cwscan.py scan"
