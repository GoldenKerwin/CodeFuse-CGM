#!/usr/bin/env bash
set -euo pipefail

# Scan files above a size threshold and report git tracking status.
# Usage: find_large_untracked.sh [threshold_mb] [scan_root]
threshold_mb="${1:-50}"
scan_root="${2:-.}"

if ! [[ "$threshold_mb" =~ ^[0-9]+$ ]]; then
  echo "threshold_mb must be an integer" >&2
  exit 2
fi

threshold_bytes=$((threshold_mb * 1024 * 1024))

while IFS= read -r -d '' file; do
  size=$(stat -c %s "$file")
  if [ "$size" -ge "$threshold_bytes" ]; then
    if git ls-files --error-unmatch "$file" >/dev/null 2>&1; then
      status="TRACKED"
    else
      status="UNTRACKED"
    fi
    printf "%s\t%s\t%s\n" "$status" "$size" "$file"
  fi
done < <(find "$scan_root" -type f -print0)
