"""Combine multiple `manifest_frames.csv` files into a single manifest (frames only)."""
import argparse
import csv
from pathlib import Path
from typing import List


def parse_args():
    p = argparse.ArgumentParser(description="Combine frame manifests")
    p.add_argument("--manifests", nargs="+", required=True, help="Input manifest CSV paths")
    p.add_argument("--output", required=True, help="Output combined manifest CSV path")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.output)
    rows = []
    fieldnames = None

    for m in args.manifests:
        p = Path(m)
        if not p.exists():
            raise FileNotFoundError(f"Manifest not found: {p}")
        with p.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            if reader.fieldnames != fieldnames:
                raise RuntimeError(f"Incompatible manifest columns: {p}")
            for r in reader:
                rows.append(r)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote combined manifest: {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
