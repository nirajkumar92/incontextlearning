#!/usr/bin/env bash
# Export the standalone source beside itself; leave no TeX build clutter there.
set -euo pipefail
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if ! command -v tectonic >/dev/null 2>&1; then
  echo "PDF export requires Tectonic. The Codex editor can still preview the source independently." >&2
  exit 1
fi
build_dir=$(mktemp -d "${TMPDIR:-/tmp}/pfn-report.XXXXXX")
trap 'rm -rf "$build_dir"' EXIT
tectonic --untrusted --keep-logs --outdir "$build_dir" "$project_root/prior_bank_assessment.tex"
cp "$build_dir/prior_bank_assessment.pdf" "$project_root/prior_bank_assessment.pdf"
printf 'Saved %s\n' "$project_root/prior_bank_assessment.pdf"
