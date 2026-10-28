
"""Inspect CICIDS2017 CSV column names without modifying any files."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "Wednesday-workingHours.pcap_ISCX.csv"


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

    # Read only the CSV header and first five rows.
    frame = pd.read_csv(CSV_PATH, nrows=5, low_memory=False)

    print("=== Training CSV ===")
    print(CSV_PATH)
    print("\n=== Exact column names ===")

    for index, column in enumerate(frame.columns, start=1):
        print(f"{index:02}. {column!r}")

    print("\n=== First five rows: selected columns ===")
    print(frame.iloc[:, : min(8, len(frame.columns))].to_string(index=False))

    print("\nInspection only: no files were modified.")


if __name__ == "__main__":
    main()
