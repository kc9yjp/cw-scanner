#!/usr/bin/env bash
# sdr-setup.sh — install / uninstall SDRangel and other SDR decoders on Arch
# (Omarchy): dump1090 (ADS-B, used by adsb-map.sh), nrsc5 (HD Radio),
# dsd-fme (digital voice) and multimon-ng. Optional: the scanners in this
# repo only need install-arch.sh.
#
#   ./sdr-setup.sh install      # repo deps, DVB blacklist, build + install AUR packages
#   ./sdr-setup.sh uninstall    # remove SDRangel, decoders + AUR libs, restore DVB driver
#   ./sdr-setup.sh uninstall --all   # also remove rtl-sdr, multimon-ng and the build cache
#   ./sdr-setup.sh status
#
# Run as your normal user (not root); sudo is called where needed.
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-$HOME/.cache/sdr-build}"   # on disk; /tmp tmpfs is too small for Qt builds
BLACKLIST=/etc/modprobe.d/blacklist-rtlsdr.conf
DVB_MODULES=(dvb_usb_rtl28xxu rtl2832 rtl2830)

REPO_PKGS=(rtl-sdr multimon-ng)   # multimon-ng: APRS (AFSK1200), CW, POCSAG, etc.
REPO_DEPS=(fftw ffmpeg4.4 lz4 qt5-base qt5-multimedia qt5-speech qt5-graphicaleffects qt5-declarative
           bladerf lighttpd    # dump1090-fa-git runtime deps, needed before makepkg
           libao               # nrsc5-git runtime dep
           codec2 mbelib lapack blas)   # dsd-fme / itpp runtime deps
BUILD_TOOLS=(git base-devel cmake)
AUR_DEPS=(qt5-websockets qt5-charts qt5-location   # dropped from official repos
          itpp)                                   # dsd-fme dep
AUR_APP=sdrangel-bin
AUR_TOOLS=(dump1090-fa-git   # ADS-B aircraft decoder (1090 MHz)
           nrsc5-git         # HD Radio (digital FM broadcast) receiver
           dsd-fme)          # DMR / D-STAR / YSF / P25 / NXDN digital voice

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
installed() { pacman -Q "$1" &>/dev/null; }

# Build an AUR package in BUILD_DIR (skips if an up-to-date package file exists).
build_aur() {
    local pkg=$1; shift
    local dir="$BUILD_DIR/$pkg"
    if [[ -d $dir/.git ]]; then
        git -C "$dir" pull -q --ff-only
    else
        git clone -q "https://aur.archlinux.org/$pkg.git" "$dir"
    fi
    (
        cd "$dir"
        if ls ./"$pkg"-[0-9]*.pkg.tar.zst &>/dev/null && [[ -z ${FORCE_REBUILD:-} ]]; then
            log "$pkg: already built, reusing"
        else
            log "$pkg: building (log: $BUILD_DIR/$pkg.log)"
            MAKEFLAGS="-j$(nproc)" makepkg -f --noconfirm "$@" >"$BUILD_DIR/$pkg.log" 2>&1 \
                || die "$pkg build failed; see $BUILD_DIR/$pkg.log"
        fi
    )
}

pkg_file() {
    ls -t "$BUILD_DIR/$1"/"$1"-[0-9]*.pkg.tar.zst | head -1
}

do_install() {
    [[ $EUID -ne 0 ]] || die "run as your normal user, not root"
    sudo -v

    log "Installing repo packages"
    sudo pacman -S --needed --noconfirm "${REPO_PKGS[@]}"
    sudo pacman -S --needed --noconfirm --asdeps "${REPO_DEPS[@]}" "${BUILD_TOOLS[@]}"

    log "Blacklisting DVB-T TV driver so SDR apps can claim the dongle"
    printf 'blacklist %s\n' "${DVB_MODULES[@]}" | sudo tee "$BLACKLIST" >/dev/null
    for m in "${DVB_MODULES[@]}"; do sudo modprobe -r "$m" 2>/dev/null || true; done

    mkdir -p "$BUILD_DIR"
    local to_install=()
    for p in "${AUR_DEPS[@]}"; do
        installed "$p" && { log "$p: already installed"; continue; }
        build_aur "$p"
        to_install+=("$(pkg_file "$p")")
    done
    if ((${#to_install[@]})); then
        log "Installing AUR libraries"
        sudo pacman -U --needed --noconfirm --asdeps "${to_install[@]}"
    fi

    build_aur "$AUR_APP"
    log "Installing $AUR_APP"
    sudo pacman -U --needed --noconfirm "$(pkg_file "$AUR_APP")"

    for p in "${AUR_TOOLS[@]}"; do
        installed "$p" && { log "$p: already installed"; continue; }
        build_aur "$p"
        log "Installing $p"
        sudo pacman -U --needed --noconfirm "$(pkg_file "$p")"
    done

    log "Done. Replug the dongle if it was plugged in before rtl-sdr was installed."
    rtl_test -t 2>&1 | head -5 || true
    log "Launch with: sdrangel"
}

do_uninstall() {
    local all=${1:-}
    [[ $EUID -ne 0 ]] || die "run as your normal user, not root"

    local remove=()
    for p in "$AUR_APP" "${AUR_TOOLS[@]}" "${AUR_DEPS[@]}"; do installed "$p" && remove+=("$p"); done
    [[ $all == --all ]] && for p in "${REPO_PKGS[@]}"; do installed "$p" && remove+=("$p"); done

    if ((${#remove[@]})); then
        log "Removing: ${remove[*]} (plus deps nothing else needs)"
        sudo pacman -Rns --noconfirm "${remove[@]}"
    else
        log "No SDR packages installed"
    fi

    if [[ -f $BLACKLIST ]]; then
        # rtl-sdr's own /usr/lib/modprobe.d/rtlsdr.conf still blacklists the
        # driver for as long as rtl-sdr stays installed.
        log "Removing extra DVB driver blacklist"
        sudo rm -f "$BLACKLIST"
        sudo modprobe dvb_usb_rtl28xxu 2>/dev/null || true
    fi

    if [[ $all == --all && -d $BUILD_DIR ]]; then
        log "Removing build cache $BUILD_DIR"
        rm -rf "$BUILD_DIR"
    fi
    log "Done."
}

do_status() {
    for p in "${REPO_PKGS[@]}" "${AUR_DEPS[@]}" "$AUR_APP" "${AUR_TOOLS[@]}"; do
        printf '  %-16s %s\n' "$p" "$(pacman -Q "$p" 2>/dev/null | cut -d' ' -f2 || true)"
    done
    printf '  %-16s %s\n' blacklist "$([[ -f $BLACKLIST ]] && echo present || echo absent)"
    printf '  %-16s %s\n' dvb-driver "$(lsmod | grep -q '^dvb_usb_rtl28xxu' && echo LOADED || echo not loaded)"
    printf '  %-16s %s\n' dongle "$(grep -l 2838 /sys/bus/usb/devices/*/idProduct &>/dev/null && echo connected || echo 'not found')"
}

case "${1:-}" in
    install)   do_install ;;
    uninstall) do_uninstall "${2:-}" ;;
    status)    do_status ;;
    *) echo "usage: $0 {install|uninstall [--all]|status}"; exit 1 ;;
esac
