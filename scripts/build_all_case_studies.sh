#!/usr/bin/env bash
# Build every case-study JSON + static plot set the viz needs for one analysis dir.
#
# Order matters: compute_geometry_primary writes geometry_primary_analysis.json,
# which the four case-study builders and generate_scatter_plots all read.
#
# Usage:
#   conda activate geopedia
#   ./scripts/build_all_case_studies.sh trained_models/layer_2/firm-sweep-3/analysis

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <analysis-dir>" >&2
    exit 1
fi

DATA_DIR="$1"

if [ ! -d "$DATA_DIR" ]; then
    echo "Not a directory: $DATA_DIR" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if ! python -c "import proteinlens" 2>/dev/null; then
    echo "proteinlens not importable — activate the geopedia conda env first." >&2
    exit 1
fi

run() {
    local label="$1"; shift
    echo
    echo "=============================================="
    echo "== $label"
    echo "=============================================="
    python "$@" --data-dir "$DATA_DIR"
}

run "geometry-primary analysis" scripts/compute_geometry_primary.py
run "subdomain case study"      scripts/build_subdomain_case_study.py
run "MEME case studies"         scripts/build_meme_case_studies.py
run "cross-family case study"   scripts/build_cross_family_case_study.py
run "NMPFam case study"         scripts/build_nmpfam_case_study.py
run "scatter plots"             scripts/generate_scatter_plots.py

echo
echo "All case studies built for: $DATA_DIR"
