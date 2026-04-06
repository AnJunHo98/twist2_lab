#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE_ROOT = _PROJECT_ROOT.parent
_DEFAULT_TRANSITION_CSV = _WORKSPACE_ROOT / "GMR" / "motion_data" / "standing_transition_full" / "transition_manifest.csv"
_DEFAULT_SOURCE_MANIFEST = (
    _PROJECT_ROOT / "source" / "twist2_lab" / "data" / "motion_data_configs" / "twist2_local_dataset.yaml"
)
_DEFAULT_OUTPUT_MANIFEST = (
    _PROJECT_ROOT / "source" / "twist2_lab" / "data" / "motion_data_configs" / "twist2_standing_transition_full.yaml"
)
_DEFAULT_OUTPUT_ROOT_PATH = "../../../../../GMR/motion_data/standing_transition_full"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a motion manifest YAML from a standing-transition CSV."
    )
    parser.add_argument(
        "--transition-csv",
        type=Path,
        default=_DEFAULT_TRANSITION_CSV,
        help="Path to transition_manifest.csv.",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=_DEFAULT_SOURCE_MANIFEST,
        help="Optional source manifest used to inherit weight/description by source_manifest_file.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=_DEFAULT_OUTPUT_MANIFEST,
        help="Path to the output manifest YAML.",
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=_DEFAULT_OUTPUT_ROOT_PATH,
        help="root_path field to write into the output manifest.",
    )
    parser.add_argument(
        "--default-weight",
        type=float,
        default=1.0,
        help="Fallback weight for motions missing from the source manifest.",
    )
    parser.add_argument(
        "--default-description",
        type=str,
        default="general movement",
        help="Fallback description for motions missing from the source manifest.",
    )
    return parser.parse_args()


def _load_source_manifest_metadata(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}

    with path.open("r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    metadata: dict[str, dict[str, Any]] = {}
    for entry in manifest.get("motions", []):
        file_key = str(entry["file"])
        metadata[file_key] = {
            "weight": float(entry.get("weight", 1.0)),
            "description": str(entry.get("description", "general movement")),
        }
    return metadata


def main() -> None:
    args = _parse_args()
    transition_csv = args.transition_csv.expanduser().resolve()
    source_manifest = args.source_manifest.expanduser().resolve()
    output_manifest = args.output_manifest.expanduser().resolve()

    source_metadata = _load_source_manifest_metadata(source_manifest)

    motions = []
    seen_files: set[str] = set()

    with transition_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            manifest_file = str(row["manifest_file"])
            if manifest_file in seen_files:
                continue
            seen_files.add(manifest_file)

            source_manifest_file = str(row.get("source_manifest_file") or manifest_file)
            inherited = source_metadata.get(source_manifest_file) or source_metadata.get(manifest_file) or {}

            motions.append(
                {
                    "file": manifest_file,
                    "weight": float(inherited.get("weight", args.default_weight)),
                    "description": str(inherited.get("description", args.default_description)),
                }
            )

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "root_path": args.root_path,
                "motions": motions,
            },
            f,
            sort_keys=False,
            allow_unicode=False,
        )

    print(f"Wrote {len(motions)} motions to: {output_manifest}")


if __name__ == "__main__":
    main()
