# Fly simulation run storage

`flysim_storage` records one behavioral MuJoCo result independently of any
camera or render style. A run is an immutable directory:

```text
walk_imitation.simrun/
  manifest.json   # schema, provenance, timing, model fingerprint, checksum
  states.npz      # compressed float64 time/qpos and available dynamic state
```

This is intentionally smaller and faster than frame-by-frame JSON. It does
not store RGB frames, joint dictionaries, or derived `xpos`/`xquat`; MuJoCo
reconstructs derived poses from `qpos` with forward kinematics.

The existing behavior CLI can write this directly. Use a fresh output
directory for each immutable run:

```powershell
$env:PYTHONPATH = "E:\Projects\bosmos_deepspace\codex"
python codex\scripts\export_flybody_behavior_suite_for_blender.py `
  --output-dir H:\Projects\bosmos_deepspace\Data\Simulations\walk_001 `
  --behaviors walk_imitation --steps 5000 --format simrun
```

The existing high-resolution style renderer can replay it without running
the controller:

```powershell
python codex\scripts\render_flybody_continuous_orbit_style_cycle.py `
  --trajectory H:\Projects\bosmos_deepspace\Data\Simulations\walk_001\walk_imitation.simrun `
  --output-dir H:\Projects\bosmos_deepspace\Renders\MuJoCo\walk_001_style_cycle `
  --fps 30 --width 3628 --height 1600 --switch-seconds 5
```

## Notebook/controller recording

Make `codex/` importable, then capture immediately after each environment or
controller step:

```python
from flysim_storage import TrajectoryRecorder

env.reset()
recorder = TrajectoryRecorder(
    env.physics,
    metadata={"behavior": "walk_imitation", "controller": "my_controller", "seed": 7},
)
recorder.append()  # optional initial state
for action in actions:
    env.step(action)
    recorder.append()
recorder.save("H:/Projects/bosmos_deepspace/Data/Simulations/walk_001.simrun")
```

Record at the controller rate. Keep float64 samples; reduce render work later
by choosing render-frame indices rather than discarding simulation state.

## Replay for any renderer

```python
from flysim_storage import TrajectoryRun

run = TrajectoryRun.load(".../walk_001.simrun")
run.validate_model(physics)  # once, before rendering
for frame_index, sample_index in enumerate(run.frame_indices(fps=30)):
    run.apply(physics, sample_index)  # sets state, then calls physics.forward()
    pixels = physics.render(width=3628, height=1600, camera_id=6)
```

The renderer may change materials, textures, lights, cameras, resolution, and
diagnostic flags. It must preserve the recorded model's kinematic layout.

## What is stored

- Required: `time`, `qpos`.
- When present: `qvel`, `act`, `ctrl`, `mocap_pos`, `mocap_quat`, `userdata`.
- Manifest: controller/seed metadata supplied by the caller, array shapes,
  sample timing, a visual-style-independent model fingerprint, and a SHA-256
  checksum of `states.npz`.

For render-only replay, `qpos` is the essential payload. The other arrays make
the run useful for diagnostics without bloating it with per-frame JSON
objects. Exact dynamic continuation is outside this format and may require
additional MuJoCo solver or plugin state.
