"""Small inspection CLI for a recorded simulation bundle."""

from __future__ import annotations

import argparse
import json

from .core import TrajectoryRun


def main():
    parser = argparse.ArgumentParser(prog="python -m flysim_storage")
    parser.add_argument("bundle", help="Path to a *.simrun directory")
    parser.add_argument("--skip-checksum", action="store_true")
    args = parser.parse_args()
    run = TrajectoryRun.load(args.bundle, verify_checksum=not args.skip_checksum)
    summary = {
        "path": str(run.path),
        "samples": run.sample_count,
        "duration_seconds": run.duration,
        "arrays": {name: list(value.shape) for name, value in sorted(run.arrays.items())},
        "model_kinematic_sha256": run.manifest["model"]["kinematic_sha256"],
        "provenance": run.manifest.get("provenance", {}),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

