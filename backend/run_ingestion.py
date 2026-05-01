"""
Run the ingestion pipeline from the command line.

Usage
-----
python run_ingestion.py data/sample_reviews.csv
python run_ingestion.py data/myfile.csv --text-column review_text
python run_ingestion.py data/myfile.csv --text-column text --no-save
"""

import argparse
import sys
import os
from pathlib import Path

# Make sure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion.pipeline import ingest
from ingestion.config import IngestionConfig
from utils.reporter import print_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Text Intelligence Platform — CSV Ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    default_file = Path(__file__).resolve().parents[1] / "data" / "sample_youtube_comments.csv"
    parser.add_argument(
        "file",
        nargs="?",
        default=str(default_file),
        help=f"Path to the CSV file to ingest (default: {default_file})",
    )
    parser.add_argument(
        "--text-column", "-t",
        default=None,
        help="Name of the text column (auto-detected if not set)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving clean CSV and JSON report",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="output",
        help="Directory to save outputs (default: output/)",
    )
    parser.add_argument(
        "--max-null-ratio",
        type=float,
        default=0.30,
        help="Error threshold for null ratio (default: 0.30)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = IngestionConfig(
        text_column=args.text_column,
        output_dir=args.output_dir,
        save_clean_csv=not args.no_save,
        save_report_json=not args.no_save,
        max_null_ratio=args.max_null_ratio,
    )

    report = ingest(args.file, config)
    print_report(report)

    sys.exit(0 if report.success and not report.has_errors else 1)


if __name__ == "__main__":
    main()
