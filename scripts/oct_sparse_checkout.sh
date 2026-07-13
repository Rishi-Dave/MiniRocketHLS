#!/usr/bin/env bash
# OCT sparse-checkout bootstrap.
# Run ON OCT after advisor provides ssh access. Pulls only the source paths
# needed for [NET]-axis work; never pulls build_dir.* / .xclbin / .xo artifacts.
#
# Usage (on OCT):
#   ./oct_sparse_checkout.sh /path/to/checkout/dir
#
# Wolverine remains source-of-truth on `master` / current branch.
# Diverged OCT-only work goes on branch `oct-net` under `fpga-network-oct/`.

set -euo pipefail

REPO_URL="https://github.com/Rishi-Dave/MiniRocketHLS.git"
BRANCH="${BRANCH:-master}"
DEST="${1:-$HOME/minirocket-hls-oct}"

PATHS=(
  "MiniRocketHLS/fpga-network"
  "MiniRocketHLS/cpu-network"
  "MiniRocketHLS/minirocket_modular/src"
  "MiniRocketHLS/minirocket_modular/host"
  "MiniRocketHLS/hydra_optimized/src"
  "MiniRocketHLS/hydra_optimized/host"
  "MiniRocketHLS/multirocket_optimized/src"
  "MiniRocketHLS/multirocket_optimized/host"
  "MiniRocketHLS/minirocket_modular/GunPoint"
  "MiniRocketHLS/scripts"
)

if [ -d "$DEST/.git" ]; then
  echo "[oct] $DEST exists; updating"
  cd "$DEST"
  git fetch origin "$BRANCH"
  git checkout "$BRANCH"
  git pull --ff-only origin "$BRANCH"
else
  echo "[oct] cloning $REPO_URL into $DEST (sparse, no-checkout)"
  git clone --filter=blob:none --no-checkout --branch "$BRANCH" "$REPO_URL" "$DEST"
  cd "$DEST"
  git sparse-checkout init --cone
fi

echo "[oct] setting sparse paths:"
printf '  %s\n' "${PATHS[@]}"
git sparse-checkout set "${PATHS[@]}"
git checkout "$BRANCH"

echo
echo "[oct] checkout summary:"
du -sh "$DEST" 2>/dev/null || true
echo "[oct] safety check — must NOT contain build_dir.* or .xclbin:"
find "$DEST" \( -name 'build_dir.*' -o -name '*.xclbin' -o -name '*.xo' \) -print | head -5 || true

echo
echo "[oct] done. Next steps on OCT:"
echo "  1. cd $DEST/MiniRocketHLS/fpga-network && vitis_hls -version"
echo "  2. csim of the existing fpga-network/minirocket kernel (N0)"
echo "  3. switch to branch oct-net before any local commits:"
echo "     git checkout -b oct-net"
