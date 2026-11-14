#!/usr/bin/env bash
# Live ADS-B aircraft map: dump1090-fa decoder + SkyAware web map on localhost.
#
#   ./adsb-map.sh                          # map centered on Chicago, auto gain
#   ./adsb-map.sh --lat 41.95 --lon -87.80 # your receiver position: site marker,
#                                          # range rings, better position decoding
#   ./adsb-map.sh --port 8090 --gain 40 --no-open
#   ./adsb-map.sh --adaptive               # auto-adjust gain vs. noise floor and loud bursts
#   ./adsb-map.sh --range-target 30 --min-gain 20 --max-gain 49.6   # (implies --adaptive)
#   ./adsb-map.sh -- --ppm 2 --fix-2bit    # anything after -- goes straight to dump1090
#
# Ctrl-C stops everything. Needs dump1090-fa-git (see sdr-setup.sh) and python3.
set -euo pipefail

HTML_SRC=/usr/share/dump1090/html
RUN_DIR="${XDG_RUNTIME_DIR:-/tmp}/adsb-map"   # tmpfs: JSON is rewritten every second
LAT='' LON='' PORT=8080 GAIN=-10 OPEN=1       # gain -10 = tuner AGC
ADAPTIVE=0 RANGE_TARGET='' MIN_GAIN='' MAX_GAIN='' EXTRA=()

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
usage() { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
is_num() { [[ $1 =~ ^-?[0-9]+(\.[0-9]+)?$ ]]; }

while (($#)); do
    case $1 in
        --lat)     LAT=${2:-}; shift 2 ;;
        --lon)     LON=${2:-}; shift 2 ;;
        --port)    PORT=${2:-}; shift 2 ;;
        --gain)    GAIN=${2:-}; shift 2 ;;
        --no-open) OPEN=0; shift ;;
        --adaptive)     ADAPTIVE=1; shift ;;
        --range-target) RANGE_TARGET=${2:-}; ADAPTIVE=1; shift 2 ;;
        --min-gain)     MIN_GAIN=${2:-}; ADAPTIVE=1; shift 2 ;;
        --max-gain)     MAX_GAIN=${2:-}; ADAPTIVE=1; shift 2 ;;
        --) shift; EXTRA=("$@"); break ;;
        -h|--help) usage ;;
        *) echo "unknown option: $1" >&2; usage 1 ;;
    esac
done

command -v dump1090 >/dev/null || die "dump1090 not found; run ./sdr-setup.sh install"
command -v python3 >/dev/null  || die "python3 not found"
[[ -d $HTML_SRC ]]             || die "$HTML_SRC missing (is dump1090-fa-git installed?)"
[[ -z $LAT$LON ]] || { is_num "$LAT" && is_num "$LON"; } || die "--lat and --lon must both be numbers"
[[ $PORT =~ ^[0-9]+$ ]]        || die "--port must be a number"
is_num "$GAIN"                 || die "--gain must be a number"
for v in RANGE_TARGET MIN_GAIN MAX_GAIN; do
    [[ -z ${!v} ]] || is_num "${!v}" || die "--$(tr _A-Z -a-z <<<"$v") must be a number"
done

# Web root: symlinks to the packaged SkyAware files, a live data/ dir, and our own config.js.
rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR/web" "$RUN_DIR/data"
for f in "$HTML_SRC"/*; do [[ ${f##*/} == data ]] || ln -s "$f" "$RUN_DIR/web/"; done
ln -s "$RUN_DIR/data" "$RUN_DIR/web/data"
rm "$RUN_DIR/web/config.js"
center_lat=${LAT:-41.90} center_lon=${LON:--87.80}
sed -e "s/^DefaultCenterLat = .*/DefaultCenterLat = $center_lat;/" \
    -e "s/^DefaultCenterLon = .*/DefaultCenterLon = $center_lon;/" \
    -e "s/^DefaultZoomLvl   = .*/DefaultZoomLvl   = 9;/" \
    "$HTML_SRC/config.js" > "$RUN_DIR/web/config.js"
if [[ -n $LAT ]]; then
    sed -i -e "s/^SiteShow    = .*/SiteShow    = true;/" \
           -e "s/^SiteLat     = .*/SiteLat     = $LAT;/" \
           -e "s/^SiteLon     = .*/SiteLon     = $LON;/" "$RUN_DIR/web/config.js"
fi

pids=()
cleanup() {
    trap - EXIT INT TERM
    ((${#pids[@]})) && kill "${pids[@]}" 2>/dev/null && wait "${pids[@]}" 2>/dev/null
    log "Stopped."
}
trap cleanup EXIT INT TERM

dump_args=(--gain "$GAIN" --net --net-bind-address 127.0.0.1
           --write-json "$RUN_DIR/data" --write-json-every 1 --quiet)
[[ -n $LAT ]] && dump_args+=(--lat "$LAT" --lon "$LON")
if ((ADAPTIVE)); then
    # --gain becomes the starting point; dump1090 then tunes it against the measured noise floor.
    dump_args+=(--adaptive-range --adaptive-burst)
    [[ -n $RANGE_TARGET ]] && dump_args+=(--adaptive-range-target "$RANGE_TARGET")
    [[ -n $MIN_GAIN ]]     && dump_args+=(--adaptive-min-gain "$MIN_GAIN")
    [[ -n $MAX_GAIN ]]     && dump_args+=(--adaptive-max-gain "$MAX_GAIN")
fi
dump_args+=("${EXTRA[@]}")

log "Starting dump1090 (log: $RUN_DIR/dump1090.log)"
dump1090 "${dump_args[@]}" >"$RUN_DIR/dump1090.log" 2>&1 &
pids+=($!)

# Fail fast if the dongle is missing or busy (e.g. SDRangel has it open).
for _ in {1..10}; do
    [[ -s $RUN_DIR/data/receiver.json ]] && break
    kill -0 "${pids[0]}" 2>/dev/null || { tail -5 "$RUN_DIR/dump1090.log" >&2; die "dump1090 exited"; }
    sleep 0.5
done
[[ -s $RUN_DIR/data/receiver.json ]] || die "dump1090 produced no data; see $RUN_DIR/dump1090.log"

log "Starting web server"
python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$RUN_DIR/web" \
    >"$RUN_DIR/http.log" 2>&1 &
pids+=($!)
sleep 0.5
kill -0 "${pids[1]}" 2>/dev/null || die "web server failed (port $PORT in use?); see $RUN_DIR/http.log"

url="http://localhost:$PORT/"
log "Map: $url   (raw feeds: 30002 raw, 30003 SBS, 30005 Beast)"
((OPEN)) && command -v xdg-open >/dev/null && xdg-open "$url" >/dev/null 2>&1 &

# Live aircraft count until Ctrl-C or either process dies.
alive() { local p; for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null || return 1; done; }
while alive; do
    n=$(python3 - "$RUN_DIR/data" <<'PY' 2>/dev/null || echo 'aircraft: ?'
import json, sys
d = sys.argv[1]
out = [f"aircraft: {len(json.load(open(d + '/aircraft.json'))['aircraft'])}"]
try:
    s = json.load(open(d + '/stats.json'))
    loc, gain = s.get('last1min', {}).get('local', {}), s.get('last1min', {}).get('adaptive', {}).get('gain_db')
    for k in ('noise', 'signal', 'peak_signal'):
        if k in loc: out.append(f"{k}: {loc[k]:.1f} dBFS")
    if 'strong_signals' in loc: out.append(f"strong: {loc['strong_signals']}")
    if gain is not None: out.append(f"gain: {gain:.1f} dB")
except (OSError, ValueError):
    pass
print('  '.join(out))
PY
)
    printf '\r\033[K%s  %s  (Ctrl-C to stop)' "$(date +%T)" "$n"
    sleep 5
done
echo
die "a process exited unexpectedly; see logs in $RUN_DIR"
