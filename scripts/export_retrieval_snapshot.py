"""Build a judged F3 snapshot from an explicit existing SQLite file, read-only."""

import argparse
from pathlib import Path

from app.evaluation.provenance import PROJECT_ROOT
from app.evaluation.snapshot_export import export_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument(
        "--query-vectors", type=Path, default=PROJECT_ROOT / "data/query_vectors.json"
    )
    parser.add_argument("--query-metadata", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metadata = args.query_metadata or args.query_vectors.with_name(
        "query_vectors.meta.json"
    )
    report = export_snapshot(
        args.request, args.database, args.query_vectors, metadata, args.output_dir
    )
    print(
        f"Exported {report['encoded_outfit_count']} outfits and {report['case_count']} cases."
    )
    print(f"Snapshot: {args.output_dir / 'snapshot.json'}")


if __name__ == "__main__":
    main()
