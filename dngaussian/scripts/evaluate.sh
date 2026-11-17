#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:?Usage: $0 baseline|depth}"
shift

case "${MODE}" in
    baseline)
        exec "${ROOT}/scripts/render_aist_heldout.sh" 5000 \
            train.tag=aist_3cam_baseline train.depth_weight=0 "$@"
        ;;
    depth)
        exec "${ROOT}/scripts/render_aist_heldout.sh" 5000 "$@"
        ;;
    *)
        echo "Usage: $0 baseline|depth" >&2
        exit 2
        ;;
esac
