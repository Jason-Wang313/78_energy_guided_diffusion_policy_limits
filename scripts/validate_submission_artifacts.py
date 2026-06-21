import csv
import hashlib
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"
DOWNLOADS_PDF = Path.home() / "Downloads" / "78.pdf"
DESKTOP_PDF = Path.home() / "Desktop" / "78.pdf"

EXPECTED_COUNTS = {
    "rollouts.csv": 6240,
    "training_summary.csv": 480,
    "raw_seed_metrics.csv": 520,
    "metrics.csv": 455,
    "pairwise_stats.csv": 60,
    "aggregate_seed_metrics.csv": 104,
    "aggregate_metrics.csv": 91,
    "aggregate_pairwise_stats.csv": 12,
    "ablation_rollouts.csv": 1600,
    "ablation_seed_metrics.csv": 160,
    "ablation_metrics.csv": 20,
    "stress_sweep_raw.csv": 5120,
    "stress_sweep.csv": 80,
    "fixed_risk_raw.csv": 2048,
    "fixed_risk_seed_metrics.csv": 256,
    "fixed_risk_metrics.csv": 192,
    "fixed_risk_pairwise.csv": 28,
    "negative_cases.csv": 12,
}

HARD_LOG_PATTERNS = [
    "Overfull",
    "Citation",
    "undefined references",
    "undefined citation",
    "Rerun to get",
    "Package natbib Warning",
    "LaTeX Warning",
]


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def count_csv(path):
    return len(read_rows(path))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def pdf_pages(path):
    out = subprocess.check_output(["pdfinfo", str(path)], text=True, stderr=subprocess.STDOUT)
    match = re.search(r"^Pages:\s+(\d+)", out, re.MULTILINE)
    if not match:
        raise AssertionError("could not read PDF page count")
    return int(match.group(1))


def main():
    for name, expected in EXPECTED_COUNTS.items():
        path = RESULTS / name
        if not path.exists():
            raise AssertionError(f"missing {path}")
        actual = count_csv(path)
        if actual != expected:
            raise AssertionError(f"{name}: expected {expected}, got {actual}")

    summary = (RESULTS / "summary.txt").read_text(encoding="utf-8")
    if "Terminal recommendation:" not in summary:
        raise AssertionError("summary missing terminal recommendation")
    if "support_aware_energy_bridge_v5" not in summary:
        raise AssertionError("summary missing v5 reference method")
    if "Fixed-risk off-support summary" not in summary:
        raise AssertionError("summary missing fixed-risk section")

    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    if "citebordercolor={0 1 0}" not in tex or "pdfborder={0 0 1.6}" not in tex:
        raise AssertionError("bright boxed citation hyperref settings missing")
    if "support_aware_energy_bridge_v5" in tex:
        raise AssertionError("raw underscore method token leaked into manuscript text")

    log_path = PAPER / "main.log"
    if not log_path.exists():
        raise AssertionError("missing LaTeX log")
    log = log_path.read_text(encoding="utf-8", errors="replace")
    for pattern in HARD_LOG_PATTERNS:
        if pattern in log:
            raise AssertionError(f"LaTeX log contains hard pattern: {pattern}")

    if not DOWNLOADS_PDF.exists():
        raise AssertionError(f"missing Downloads PDF: {DOWNLOADS_PDF}")
    if DESKTOP_PDF.exists():
        raise AssertionError(f"Desktop PDF must not exist: {DESKTOP_PDF}")
    pages = pdf_pages(DOWNLOADS_PDF)
    if pages < 25:
        raise AssertionError(f"PDF must be at least 25 pages, got {pages}")

    print(f"validated Paper 78 artifacts: pages={pages}, sha256={sha256(DOWNLOADS_PDF)}")


if __name__ == "__main__":
    main()
