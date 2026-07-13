#!/usr/bin/env python3
"""
Extract FPGA resource utilization from Vivado and HLS synthesis report files.

Searches the MiniRocketHLS project tree and parses:
  - Vivado post-route utilization reports (*_utilization_routed.rpt)
  - Vivado post-place utilization reports (*_utilization_placed.rpt)
  - HLS synthesis summary reports (csynth.rpt) — top-level kernel row only

Outputs: results/resource_utilization.csv

U280 total resources (used as fallback when Available column is absent):
  CLB LUT:        1,303,680
  CLB Registers:  2,607,360
  Block RAM Tile:     2,016  (RAMB36 tile count; RAMB18 = 4,032)
  DSP:                9,024
  URAM:                 960
"""

import os
import re
import csv
import sys

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# U280 total resources (fallback when report omits Available)
U280 = {
    "LUT":  1_303_680,
    "FF":   2_607_360,
    "BRAM": 2_016,
    "DSP":  9_024,
    "URAM": 960,
}

OUTPUT_CSV = os.path.join(PROJECT_ROOT, "results", "resource_utilization.csv")

CSV_HEADER = [
    "Variant", "Report_Type",
    "LUT_Used", "LUT_Avail", "LUT_Pct",
    "FF_Used",  "FF_Avail",  "FF_Pct",
    "BRAM_Used","BRAM_Avail","BRAM_Pct",
    "DSP_Used", "DSP_Avail", "DSP_Pct",
    "URAM_Used","URAM_Avail","URAM_Pct",
    "Source_File",
]

# ---------------------------------------------------------------------------
# Helper: convert raw table-cell string to float
# ---------------------------------------------------------------------------

def _to_float(s):
    """Return float from a raw cell string, or None on failure."""
    s = str(s).strip().replace(",", "")
    if not s or s in ("-", "~0%"):
        return None
    s = s.lstrip("<>")
    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Parser 1: Vivado utilization report
#
# Table row format:
#   | Site Type                | Used | Fixed | Prohibited | Available | Util% |
#
# Target rows: CLB LUTs, CLB Registers, Block RAM Tile, DSPs, URAM
# ---------------------------------------------------------------------------

# Match a table data row (not a separator)
_TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")
_SEPARATOR_RE = re.compile(r"^\s*\+[-+]+\+\s*$")


def _parse_table_cells(line):
    """Return list of stripped cell strings, or None if not a data row."""
    if _SEPARATOR_RE.match(line):
        return None
    m = _TABLE_ROW_RE.match(line)
    if not m:
        return None
    return [c.strip() for c in m.group(1).split("|")]


def parse_vivado_utilization(path):
    """
    Parse a Vivado utilization .rpt file.

    Returns dict with keys lut_used/lut_avail/lut_pct, ff_*, bram_*, dsp_*, uram_*
    or None if the file does not look like a utilization report.
    """
    result = {}
    try:
        with open(path, errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return None

    for line in lines:
        cells = _parse_table_cells(line)
        if not cells or len(cells) < 2:
            continue

        # Standard Vivado columns: Site Type(0) | Used(1) | Fixed(2) | Prohibited(3) | Available(4) | Util%(5)
        if len(cells) < 5:
            continue

        site = re.sub(r"\*.*$", "", cells[0]).strip()
        used  = _to_float(cells[1])
        avail = _to_float(cells[4]) if len(cells) >= 5 else None
        pct   = _to_float(cells[5]) if len(cells) >= 6 else None

        if used is None:
            continue

        # CLB LUTs — first-match wins (avoids sub-type rows)
        if re.match(r"^CLB LUTs?$", site, re.IGNORECASE) and "lut_used" not in result:
            result["lut_used"]  = int(used)
            result["lut_avail"] = int(avail) if avail else U280["LUT"]
            result["lut_pct"]   = pct

        # CLB Registers
        elif re.match(r"^CLB Registers?$", site, re.IGNORECASE) and "ff_used" not in result:
            result["ff_used"]  = int(used)
            result["ff_avail"] = int(avail) if avail else U280["FF"]
            result["ff_pct"]   = pct

        # Block RAM Tile (RAMB36-equivalent tile count)
        elif re.match(r"^Block RAM Tile$", site, re.IGNORECASE) and "bram_used" not in result:
            result["bram_used"]  = used          # keep as float (can be fractional in synth rpts)
            result["bram_avail"] = int(avail) if avail else U280["BRAM"]
            result["bram_pct"]   = pct

        # DSPs
        elif re.match(r"^DSPs?$", site, re.IGNORECASE) and "dsp_used" not in result:
            result["dsp_used"]  = int(used)
            result["dsp_avail"] = int(avail) if avail else U280["DSP"]
            result["dsp_pct"]   = pct

        # URAM
        elif re.match(r"^URAM$", site, re.IGNORECASE) and "uram_used" not in result:
            result["uram_used"]  = int(used)
            result["uram_avail"] = int(avail) if avail else U280["URAM"]
            result["uram_pct"]   = pct

    # Only return if we found at least LUT or BRAM data (otherwise probably not a util rpt)
    if "lut_used" not in result and "bram_used" not in result:
        return None

    # Fill missing URAM with zeros (many designs have 0 URAM)
    if "uram_used" not in result:
        result["uram_used"]  = 0
        result["uram_avail"] = U280["URAM"]
        result["uram_pct"]   = 0.0

    return result


# ---------------------------------------------------------------------------
# Parser 2: HLS csynth.rpt
#
# The Performance & Resource Estimates table has a header row:
#   | BRAM | DSP | FF | LUT | URAM |
#
# The first top-level module row (starts with "|+") has values like:
#   185 (4%)   or   724 (8%)   or   -
#
# We extract the top-level module row only.
# ---------------------------------------------------------------------------

_CSYNTH_VALUE_RE = re.compile(r"(\d[\d,]*)\s*\(\s*([^)]+)\s*\)")


def _parse_csynth_cell(cell):
    """Return (used: int, pct: float) or (None, None)."""
    m = _CSYNTH_VALUE_RE.search(cell)
    if not m:
        return None, None
    used = int(m.group(1).replace(",", ""))
    pct_str = m.group(2).strip().rstrip("%").replace("~", "")
    pct = _to_float(pct_str)
    return used, pct


def parse_csynth(path):
    """
    Parse an HLS csynth.rpt and extract top-level module resource estimates.
    Returns dict with same keys as parse_vivado_utilization, or None.
    """
    try:
        with open(path, errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return None

    in_perf_section = False
    header_found = False
    col_bram = col_dsp = col_ff = col_lut = col_uram = None

    for line in lines:
        stripped = line.strip()

        # Enter Performance & Resource Estimates section
        if "Performance & Resource Estimates" in stripped:
            in_perf_section = True
            continue

        if not in_perf_section:
            continue

        # Exit on next major section (== heading) or double-bar divider
        if stripped.startswith("================================================================"):
            if header_found:
                break
            continue

        if "|" not in line:
            continue

        parts = line.split("|")

        # Find header row containing BRAM, DSP, FF, LUT, URAM
        if not header_found and "BRAM" in line and "LUT" in line and "FF" in line:
            for i, p in enumerate(parts):
                ps = p.strip()
                if ps == "BRAM":
                    col_bram = i
                elif ps == "DSP":
                    col_dsp = i
                elif ps == "FF":
                    col_ff = i
                elif ps == "LUT":
                    col_lut = i
                elif ps == "URAM":
                    col_uram = i
            header_found = True
            continue

        if not header_found:
            continue

        # Top-level module row: the module name cell starts with "+"
        # (after the leading "|")
        if len(parts) < 10:
            continue
        name_cell = parts[1] if len(parts) > 1 else ""
        if not re.match(r"\s*\+", name_cell):
            continue

        # Parse resource columns
        result = {}

        def extract_col(col, res_name, fallback_avail):
            if col is not None and col < len(parts):
                used, pct = _parse_csynth_cell(parts[col])
                if used is not None:
                    result[f"{res_name}_used"]  = used
                    result[f"{res_name}_avail"] = fallback_avail
                    result[f"{res_name}_pct"]   = pct if pct is not None else round(
                        used / fallback_avail * 100, 2
                    )

        extract_col(col_bram, "bram", U280["BRAM"])
        extract_col(col_dsp,  "dsp",  U280["DSP"])
        extract_col(col_ff,   "ff",   U280["FF"])
        extract_col(col_lut,  "lut",  U280["LUT"])

        # URAM: often "-" — treat as 0 if absent
        uram_used, uram_pct = None, None
        if col_uram is not None and col_uram < len(parts):
            uram_used, uram_pct = _parse_csynth_cell(parts[col_uram])
        result["uram_used"]  = uram_used if uram_used is not None else 0
        result["uram_avail"] = U280["URAM"]
        result["uram_pct"]   = (uram_pct if uram_pct is not None
                                else round(result["uram_used"] / U280["URAM"] * 100, 2))

        if result:
            return result

    return None


# ---------------------------------------------------------------------------
# Variant name inference from file path
# ---------------------------------------------------------------------------

def infer_variant(path):
    """
    Derive a human-readable variant label from the report file path.
    Checks known directory patterns starting from PROJECT_ROOT.
    """
    rel = path[len(PROJECT_ROOT):].lstrip(os.sep)
    parts = rel.split(os.sep)
    top = parts[0] if parts else "unknown"

    if top == "minirocket_modular":
        # Look for a build identifier in the path
        path_lower = rel.lower()
        if "fused_3cu" in path_lower:
            return "modular_fused_3cu"
        if "fused_2cu" in path_lower:
            return "modular_fused_2cu"
        if re.search(r"fused(?!_\d)", path_lower):
            return "modular_fused_1cu"
        if "v16_fixed" in path_lower:
            return "modular_v16_fixed"
        if re.search(r"v16\b", path_lower):
            return "modular_v16"
        # Check for sub-kernel name (scaler, classifier, feature_extraction)
        for p in parts:
            if "scaler" in p.lower():
                return "modular_baseline_scaler"
            if "classifier" in p.lower():
                return "modular_baseline_classifier"
            if "feature_extraction" in p.lower():
                return "modular_baseline_fe"
        return "modular_baseline"

    if top == "hydra_optimized":
        path_lower = rel.lower()
        if "u16_fixed" in path_lower or re.search(r"hw\.u16", path_lower):
            return "hydra_v2_apfixed"
        if "280mhz_fixed" in path_lower:
            return "hydra_280mhz_fixed"
        # Check which build dir
        for p in parts:
            if p.startswith("build"):
                return "hydra_v2_apfixed"
        return "hydra_optimized"

    if top == "reference_1to1":
        return "reference_1to1"

    if top == "optimized_version":
        return "optimized_version"

    if top == "minirocket_stream":
        # sub-kernel
        for p in parts:
            pl = p.lower()
            if "load" in pl:
                return "stream_load"
            if "store" in pl:
                return "stream_store"
            if "inference" in pl or "minirocket" in pl:
                return "stream_inference"
        return "minirocket_stream"

    if top == "multirocket_optimized":
        return "multirocket_optimized"

    # fallback
    return top


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

# Directories that are just redundant copies — skip to avoid duplicates
_SKIP_DIRS = {
    "hls_files",     # HLS IP copies inside .xo / impl directories
    "__pycache__",
    ".git",
}

# Path fragment patterns that indicate a file is a per-block sub-module synth
# report (not the full system), and should be skipped when we already have
# a routed report.
_SKIP_SYNTH_PATTERNS = [
    r"/prj\.runs/",          # per-block Vivado runs inside a full system build
]


def _should_skip_dir(dirname, dirpath_rel):
    if dirname in _SKIP_DIRS:
        return True
    return False


def find_reports():
    """
    Walk the project tree and collect relevant .rpt files.
    Returns list of (abs_path, report_type) tuples.
    """
    reports = []
    seen = set()

    for dirpath, dirnames, filenames in os.walk(PROJECT_ROOT):
        # Prune unwanted directories in-place
        dirnames[:] = [
            d for d in dirnames
            if not _should_skip_dir(d, dirpath)
        ]

        # Limit depth to avoid extremely deep trees (e.g., inside Vivado project.runs)
        rel = dirpath[len(PROJECT_ROOT):].lstrip(os.sep)
        depth = rel.count(os.sep)
        if depth > 14:
            dirnames.clear()
            continue

        for fname in filenames:
            fpath = os.path.join(dirpath, fname)

            # --- Vivado post-route utilization reports ---
            if fname.endswith("_utilization_routed.rpt"):
                if "clock_utilization" in fname:
                    continue
                if fpath not in seen:
                    seen.add(fpath)
                    reports.append((fpath, "vivado_routed"))

            # --- Vivado post-place utilization reports (fallback) ---
            elif fname.endswith("_utilization_placed.rpt"):
                if "clock" in fname:
                    continue
                if fpath not in seen:
                    seen.add(fpath)
                    reports.append((fpath, "vivado_placed"))

            # --- HLS csynth summary reports ---
            elif fname == "csynth.rpt":
                if fpath not in seen:
                    seen.add(fpath)
                    reports.append((fpath, "hls_csynth"))

    return reports


# ---------------------------------------------------------------------------
# Row construction and deduplication
# ---------------------------------------------------------------------------

def _fmt(val, default=""):
    """Format a value for CSV output."""
    if val is None:
        return default
    if isinstance(val, float) and val == int(val):
        return int(val)
    return val


def build_row(variant, report_type, data, source_file):
    """Return a dict suitable for writing to CSV."""
    return {
        "Variant":     variant,
        "Report_Type": report_type,
        "LUT_Used":    _fmt(data.get("lut_used")),
        "LUT_Avail":   _fmt(data.get("lut_avail")),
        "LUT_Pct":     _fmt(data.get("lut_pct")),
        "FF_Used":     _fmt(data.get("ff_used")),
        "FF_Avail":    _fmt(data.get("ff_avail")),
        "FF_Pct":      _fmt(data.get("ff_pct")),
        "BRAM_Used":   _fmt(data.get("bram_used")),
        "BRAM_Avail":  _fmt(data.get("bram_avail")),
        "BRAM_Pct":    _fmt(data.get("bram_pct")),
        "DSP_Used":    _fmt(data.get("dsp_used")),
        "DSP_Avail":   _fmt(data.get("dsp_avail")),
        "DSP_Pct":     _fmt(data.get("dsp_pct")),
        "URAM_Used":   _fmt(data.get("uram_used")),
        "URAM_Avail":  _fmt(data.get("uram_avail")),
        "URAM_Pct":    _fmt(data.get("uram_pct")),
        "Source_File": source_file,
    }


def _row_score(row):
    """Count non-empty numeric fields — used to prefer richer reports."""
    return sum(1 for c in CSV_HEADER[2:-1] if row.get(c) not in ("", None))


def deduplicate(rows):
    """
    For each (Variant, Report_Type) pair keep the most data-rich row.
    Then sort by variant name and report type priority.
    """
    priority = {"vivado_routed": 0, "vivado_placed": 1, "hls_csynth": 2}
    best = {}
    for row in rows:
        key = (row["Variant"], row["Report_Type"])
        if key not in best or _row_score(row) > _row_score(best[key]):
            best[key] = row

    return sorted(
        best.values(),
        key=lambda r: (r["Variant"], priority.get(r["Report_Type"], 99)),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    reports = find_reports()
    print(f"Found {len(reports)} candidate report files.")

    rows = []
    skipped = 0

    for path, rtype in reports:
        if rtype in ("vivado_routed", "vivado_placed"):
            data = parse_vivado_utilization(path)
        elif rtype == "hls_csynth":
            data = parse_csynth(path)
        else:
            data = None

        if not data:
            skipped += 1
            continue

        variant  = infer_variant(path)
        rel_path = path[len(PROJECT_ROOT):].lstrip(os.sep)
        row = build_row(variant, rtype, data, rel_path)
        rows.append(row)

    print(f"Parsed {len(rows)} rows successfully, skipped {skipped} files with no data.")

    rows = deduplicate(rows)
    print(f"After deduplication: {len(rows)} unique (Variant, Report_Type) rows.")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nOutput: {OUTPUT_CSV}")

    # Print summary table
    print()
    hdr = (f"{'Variant':<35} {'Type':<18} {'LUT_Used':>9} {'LUT%':>6} "
           f"{'FF_Used':>9} {'FF%':>6} {'BRAM':>6} {'DSP':>6} {'URAM':>5}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        lut_pct  = r.get("LUT_Pct",  "")
        ff_pct   = r.get("FF_Pct",   "")
        lut_pct_s  = f"{lut_pct:.2f}" if isinstance(lut_pct, float) else str(lut_pct)
        ff_pct_s   = f"{ff_pct:.2f}"  if isinstance(ff_pct,  float) else str(ff_pct)
        print(
            f"{r['Variant']:<35} {r['Report_Type']:<18} "
            f"{str(r.get('LUT_Used','')):>9} {lut_pct_s:>6}% "
            f"{str(r.get('FF_Used','')):>9} {ff_pct_s:>6}% "
            f"{str(r.get('BRAM_Used','')):>6} "
            f"{str(r.get('DSP_Used','')):>6} "
            f"{str(r.get('URAM_Used','')):>5}"
        )


if __name__ == "__main__":
    main()
