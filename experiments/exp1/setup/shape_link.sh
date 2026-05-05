#!/usr/bin/env bash
# shape_link.sh — apply/remove tc/netem shaping on a Chameleon client node
#
# Usage:
#   sudo bash shape_link.sh apply                # 10 Mbps cap
#   sudo bash shape_link.sh apply --jittery      # 10 Mbps + 30% bw jitter + 2% loss
#   sudo bash shape_link.sh remove --jittery     # drop netem overlay only (TBF stays)
#   sudo bash shape_link.sh remove               # drop ALL shaping
#   sudo bash shape_link.sh status               # show current qdiscs
#
# Auto-detects the interface — tries eno1np0, then eno1, then whichever
# device holds the default route.

set -euo pipefail

# --- interface detection ---
detect_iface() {
    for cand in eno1np0 eno1; do
        if ip link show "$cand" &>/dev/null; then
            echo "$cand"
            return 0
        fi
    done
    local fallback
    fallback=$(ip route | awk '/^default/ {print $5; exit}')
    if [[ -n "$fallback" ]] && ip link show "$fallback" &>/dev/null; then
        echo "$fallback"
        return 0
    fi
    return 1
}

usage() {
    cat <<EOF
Usage: sudo bash $0 <command> [--jittery]

Commands:
  apply               install 10 Mbps TBF cap
  apply --jittery     same plus netem overlay (30% bw jitter + 2% loss)
  remove --jittery    drop ONLY the netem overlay (TBF stays in place)
  remove              drop ALL shaping (clears the entire qdisc tree)
  status              print the current qdisc tree on the detected interface
EOF
}

# --- prerequisites ---
if [[ "${EUID}" -ne 0 ]]; then
    echo "ERROR: must run as root (use sudo)" >&2
    exit 1
fi

CMD="${1:-}"
shift || true

JITTERY=0
for arg in "$@"; do
    case "$arg" in
        --jittery) JITTERY=1 ;;
        *) echo "unknown flag: $arg" >&2; usage; exit 2 ;;
    esac
done

IFACE=$(detect_iface) || {
    echo "ERROR: could not find a candidate interface (tried eno1np0, eno1, default route)" >&2
    echo "Available interfaces:" >&2
    ip -br link show >&2
    exit 1
}
echo "[shape_link] interface: $IFACE"

# --- main ---
case "$CMD" in
    apply)
        echo "[shape_link] clearing any existing qdisc..."
        tc qdisc del dev "$IFACE" root 2>/dev/null || true

        echo "[shape_link] applying 10 Mbps TBF cap..."
        tc qdisc add dev "$IFACE" root handle 1: tbf \
            rate 10mbit burst 32kbit latency 50ms

        if [[ "$JITTERY" -eq 1 ]]; then
            echo "[shape_link] adding jittery netem overlay (30% bw, 2% loss)..."
            tc qdisc add dev "$IFACE" parent 1:1 handle 10: netem \
                rate 10mbit 30% loss 2%
        fi

        echo
        echo "[shape_link] active qdiscs on $IFACE:"
        tc -s qdisc show dev "$IFACE"
        echo
        echo "[shape_link] ✓ shaping applied ($([ $JITTERY -eq 1 ] && echo JITTERY || echo PRIMARY))"
        ;;

    remove)
        if [[ "$JITTERY" -eq 1 ]]; then
            # Targeted removal: drop only the netem overlay; leave TBF intact.
            if tc qdisc show dev "$IFACE" | grep -q 'qdisc netem'; then
                tc qdisc del dev "$IFACE" parent 1:1 handle 10: 2>/dev/null || true
                echo "[shape_link] ✓ removed netem overlay (TBF retained) on $IFACE"
            else
                echo "[shape_link] no netem overlay present on $IFACE — nothing to remove"
            fi
            echo
            echo "[shape_link] remaining qdiscs on $IFACE:"
            tc -s qdisc show dev "$IFACE"
        else
            # Full teardown: drop the whole qdisc tree.
            tc qdisc del dev "$IFACE" root 2>/dev/null || true
            echo "[shape_link] ✓ all shaping removed from $IFACE"
            tc qdisc show dev "$IFACE"
        fi
        ;;

    status)
        tc -s qdisc show dev "$IFACE"
        ;;

    ""|help|-h|--help)
        usage
        ;;

    *)
        echo "unknown command: $CMD" >&2
        usage
        exit 2
        ;;
esac