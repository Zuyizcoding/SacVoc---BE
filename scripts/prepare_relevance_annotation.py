"""Prepare blinded F3 relevance forms from profiles and an explicit SQLite catalog."""

import argparse
from pathlib import Path

from app.evaluation.annotation_packet import prepare_annotation_packet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reviewer-one", default="rater-a")
    parser.add_argument("--reviewer-two", default="rater-b")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    packet = prepare_annotation_packet(
        args.plan,
        args.database,
        args.output_dir,
        reviewer_one=args.reviewer_one,
        reviewer_two=args.reviewer_two,
        seed=args.seed,
    )
    print(f"Prepared {len(packet.tasks)} tasks per reviewer.")
    print(
        "Keep private/ with the coordinator; share only the assigned reviewer folder."
    )


if __name__ == "__main__":
    main()
