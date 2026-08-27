"""Collaborator template for saving a reusable MuJoCo simulation run.

Copy the simulation loop into ``run_simulation``.  The recorder captures exact
float64 MuJoCo state (qpos plus available dynamic arrays), writes an immutable
``.simrun`` bundle, and refuses to overwrite an existing run.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import mujoco

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "codex") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "codex"))

from flysim_storage import TrajectoryRecorder  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_simulation(model: mujoco.MjModel, data: mujoco.MjData, recorder: TrajectoryRecorder, steps: int) -> None:
    """Replace the marked controller section with the collaborator's model loop."""
    for _ in range(steps):
        # Set data.ctrl here, or call the collaborator's controller here.
        # Example: data.ctrl[:] = controller(data)
        mujoco.mj_step(model, data)
        recorder.append()  # capture only after the complete step


def main() -> None:
    parser = argparse.ArgumentParser(description="Save an immutable MuJoCo .simrun bundle")
    parser.add_argument("--model-xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory ending in .simrun")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--controller", default="collaborator-controller")
    args = parser.parse_args()
    model_xml = args.model_xml.resolve()
    output = args.output.resolve()
    if not model_xml.is_file():
        raise FileNotFoundError(model_xml)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable run: {output}")
    if args.steps < 1:
        raise ValueError("--steps must be positive")

    model = mujoco.MjModel.from_xml_path(str(model_xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    recorder = TrajectoryRecorder(data, metadata={
        "run_name": args.run_name,
        "generator": "save_mujoco_simrun_template.py",
        "model_xml": str(model_xml),
        "model_xml_sha256": sha256_file(model_xml),
        "controller": args.controller,
        "notes": "Replace run_simulation with the collaborator's deterministic controller loop.",
    })
    run_simulation(model, data, recorder, args.steps)
    saved = recorder.save(output)
    print(f"Saved immutable MuJoCo run: {saved}")
    print(f"Manifest: {saved / 'manifest.json'}")
    print(f"States:   {saved / 'states.npz'}")


if __name__ == "__main__":
    main()
