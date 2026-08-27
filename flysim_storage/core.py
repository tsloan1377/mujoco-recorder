"""Write and replay compact MuJoCo simulation run bundles.

The bundle deliberately stores simulation state, not pixels or derived body
poses. A renderer loads the same kinematic model, applies one recorded qpos
sample, calls forward kinematics, and is then free to change cameras,
materials, textures, lighting, and output resolution.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SCHEMA_NAME = "bosmos.mujoco-trajectory"
SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
STATES_NAME = "states.npz"


class ModelMismatchError(ValueError):
    """Raised when a run is replayed against a different kinematic model."""


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_names(model, count_name, object_kind):
    count = int(getattr(model, count_name, 0))
    names = []
    for index in range(count):
        try:
            name = model.id2name(index, object_kind)
        except (AttributeError, TypeError, ValueError):
            # dm_control exposes ``id2name`` on its model wrapper, while the
            # official mujoco Python bindings expose the equivalent C API.
            # Supporting both keeps the compact trajectory format renderer-
            # agnostic and lets tutorial notebooks record raw MjModel/MjData.
            try:
                import mujoco

                object_types = {
                    "body": mujoco.mjtObj.mjOBJ_BODY,
                    "joint": mujoco.mjtObj.mjOBJ_JOINT,
                    "actuator": mujoco.mjtObj.mjOBJ_ACTUATOR,
                }
                name = mujoco.mj_id2name(model, object_types[object_kind], index)
            except (ImportError, KeyError, TypeError, ValueError):
                name = None
        names.append(name or "")
    return names


def _hash_array(digest, name, value):
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(name.encode("utf-8"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(array.tobytes(order="C"))


def model_signature(model):
    """Return a stable kinematic signature while ignoring visual styling.

    This lets replay use different material and lighting settings while still
    rejecting a model whose generalized-coordinate layout is incompatible.
    """

    dimension_names = (
        "nq",
        "nv",
        "na",
        "nu",
        "nbody",
        "njnt",
        "nmocap",
        "nuserdata",
    )
    dimensions = {name: int(getattr(model, name, 0)) for name in dimension_names}
    names = {
        "bodies": _model_names(model, "nbody", "body"),
        "joints": _model_names(model, "njnt", "joint"),
        "actuators": _model_names(model, "nu", "actuator"),
    }

    digest = hashlib.sha256()
    digest.update(json.dumps(dimensions, sort_keys=True).encode("utf-8"))
    digest.update(json.dumps(names, sort_keys=True).encode("utf-8"))
    for attr in (
        "qpos0",
        "body_parentid",
        "jnt_type",
        "jnt_bodyid",
        "jnt_qposadr",
        "jnt_dofadr",
        "jnt_axis",
        "jnt_pos",
    ):
        if hasattr(model, attr):
            _hash_array(digest, attr, getattr(model, attr))

    return {
        "kinematic_sha256": digest.hexdigest(),
        "dimensions": dimensions,
        "names": names,
    }


def _data_from_physics(physics_or_data):
    return getattr(physics_or_data, "data", physics_or_data)


def _model_from_physics(physics_or_model):
    return getattr(physics_or_model, "model", physics_or_model)


def _copy_if_present(data, name):
    if not hasattr(data, name):
        return None
    value = np.asarray(getattr(data, name))
    if value.size == 0:
        return None
    return np.array(value, dtype=np.float64, copy=True)


class TrajectoryRecorder:
    """Accumulate exact float64 MuJoCo samples and write one run bundle.

    qpos is always recorded. qvel, actuator state, controls, mocap state, and
    userdata are included when the model exposes non-empty arrays. Derived
    xpos/xquat data is omitted because MuJoCo regenerates it with mj_forward.
    """

    OPTIONAL_FIELDS = (
        "qvel",
        "act",
        "ctrl",
        "mocap_pos",
        "mocap_quat",
        "userdata",
    )

    def __init__(self, physics, metadata=None):
        self.physics = physics
        self.model = _model_from_physics(physics)
        self.metadata = dict(metadata or {})
        self._samples = {"time": [], "qpos": []}
        data = _data_from_physics(physics)
        for name in self.OPTIONAL_FIELDS:
            if _copy_if_present(data, name) is not None:
                self._samples[name] = []

    def append(self, time_seconds=None):
        """Capture the current MuJoCo state after a controller/environment step."""

        data = _data_from_physics(self.physics)
        qpos = _copy_if_present(data, "qpos")
        if qpos is None:
            raise ValueError("MuJoCo data has no non-empty qpos array")

        if time_seconds is None:
            if not hasattr(data, "time"):
                raise ValueError("time_seconds is required when data.time is unavailable")
            time_seconds = float(data.time)

        time_seconds = float(time_seconds)
        if self._samples["time"] and time_seconds < self._samples["time"][-1]:
            raise ValueError("trajectory time must be monotonically non-decreasing")

        self._samples["time"].append(time_seconds)
        self._samples["qpos"].append(qpos)
        for name in self.OPTIONAL_FIELDS:
            if name not in self._samples:
                continue
            value = _copy_if_present(data, name)
            if value is None:
                raise ValueError("state field disappeared while recording: {}".format(name))
            self._samples[name].append(value)
        return len(self._samples["time"]) - 1

    @property
    def sample_count(self):
        return len(self._samples["time"])

    def arrays(self):
        if not self.sample_count:
            raise ValueError("cannot save an empty trajectory")
        arrays = {"time": np.asarray(self._samples["time"], dtype=np.float64)}
        for name, samples in self._samples.items():
            if name == "time":
                continue
            arrays[name] = np.stack(samples).astype(np.float64, copy=False)
        return arrays

    def save(self, output_dir):
        """Atomically create output_dir/{manifest.json,states.npz}.

        Existing bundles are never overwritten; use a new run directory so a
        render can always refer back to an immutable simulation result.
        """

        output_dir = Path(output_dir)
        if output_dir.exists():
            raise FileExistsError("run bundle already exists: {}".format(output_dir))
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix=output_dir.name + ".tmp-", dir=str(output_dir.parent)))
        try:
            arrays = self.arrays()
            states_path = temp_dir / STATES_NAME
            np.savez_compressed(str(states_path), **arrays)
            times = arrays["time"]
            deltas = np.diff(times)
            positive_deltas = deltas[deltas > 0]
            nominal_dt = float(np.median(positive_deltas)) if positive_deltas.size else None
            array_manifest = {
                name: {"dtype": str(value.dtype), "shape": list(value.shape)}
                for name, value in sorted(arrays.items())
            }
            manifest = {
                "schema": SCHEMA_NAME,
                "schema_version": SCHEMA_VERSION,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "data_file": STATES_NAME,
                "data_sha256": _sha256_file(states_path),
                "trajectory": {
                    "sample_count": int(times.size),
                    "start_time_seconds": float(times[0]),
                    "end_time_seconds": float(times[-1]),
                    "duration_seconds": float(times[-1] - times[0]),
                    "nominal_sample_timestep_seconds": nominal_dt,
                    "arrays": array_manifest,
                    "replay_basis": "qpos + MuJoCo forward kinematics",
                    "quaternion_convention": "MuJoCo WXYZ where present in qpos/mocap_quat",
                },
                "model": model_signature(self.model),
                "provenance": self.metadata,
            }
            manifest_path = temp_dir / MANIFEST_NAME
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(str(temp_dir), str(output_dir))
        except Exception:
            shutil.rmtree(str(temp_dir), ignore_errors=True)
            raise
        return output_dir


class TrajectoryRun:
    """Loaded immutable simulation result with exact-sample replay helpers."""

    def __init__(self, path, manifest, arrays):
        self.path = Path(path)
        self.manifest = manifest
        self.arrays = arrays

    @classmethod
    def load(cls, path, verify_checksum=True):
        path = Path(path)
        manifest = json.loads((path / MANIFEST_NAME).read_text(encoding="utf-8"))
        if manifest.get("schema") != SCHEMA_NAME or manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported simulation bundle schema")
        states_path = path / manifest["data_file"]
        if verify_checksum and _sha256_file(states_path) != manifest.get("data_sha256"):
            raise ValueError("simulation state checksum does not match manifest")
        with np.load(str(states_path), allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
        expected_count = int(manifest["trajectory"]["sample_count"])
        if arrays.get("time") is None or arrays.get("qpos") is None:
            raise ValueError("simulation bundle must contain time and qpos")
        if any(value.shape[0] != expected_count for value in arrays.values()):
            raise ValueError("simulation arrays do not share the manifest sample count")
        return cls(path, manifest, arrays)

    @property
    def sample_count(self):
        return int(self.arrays["time"].shape[0])

    @property
    def start_time(self):
        return float(self.arrays["time"][0])

    @property
    def duration(self):
        return float(self.arrays["time"][-1] - self.arrays["time"][0])

    def validate_model(self, physics_or_model):
        actual = model_signature(_model_from_physics(physics_or_model))
        expected = self.manifest["model"]
        if actual["kinematic_sha256"] != expected["kinematic_sha256"]:
            raise ModelMismatchError(
                "run/model kinematic fingerprints differ (run {}, model {})".format(
                    expected["kinematic_sha256"], actual["kinematic_sha256"]
                )
            )
        return True

    def index_at(self, time_seconds, mode="nearest"):
        times = self.arrays["time"]
        target = float(time_seconds)
        right = int(np.searchsorted(times, target, side="left"))
        if mode == "floor":
            return max(0, min(len(times) - 1, right if right < len(times) and times[right] == target else right - 1))
        if mode != "nearest":
            raise ValueError("mode must be 'nearest' or 'floor'")
        if right <= 0:
            return 0
        if right >= len(times):
            return len(times) - 1
        left = right - 1
        return left if target - times[left] <= times[right] - target else right

    def frame_indices(self, fps, duration_seconds=None):
        """Map constant-rate render frames to exact recorded samples."""

        fps = float(fps)
        if fps <= 0:
            raise ValueError("fps must be positive")
        duration = self.duration if duration_seconds is None else min(float(duration_seconds), self.duration)
        count = max(1, int(np.ceil(max(0.0, duration) * fps)))
        targets = self.start_time + np.arange(count, dtype=np.float64) / fps
        times = self.arrays["time"]
        right = np.searchsorted(times, targets, side="left")
        right = np.clip(right, 0, len(times) - 1)
        left = np.clip(right - 1, 0, len(times) - 1)
        choose_left = np.abs(targets - times[left]) <= np.abs(times[right] - targets)
        return np.where(choose_left, left, right).astype(np.int64)

    def apply(self, physics, index, forward=True):
        """Apply an exact stored sample to dm_control Physics or mujoco.MjData."""

        index = int(index)
        if index < 0 or index >= self.sample_count:
            raise IndexError(index)
        data = _data_from_physics(physics)
        for name, values in self.arrays.items():
            if name == "time" or not hasattr(data, name):
                continue
            target = np.asarray(getattr(data, name))
            sample = values[index]
            if target.shape != sample.shape:
                raise ModelMismatchError(
                    "state field {} has shape {}, expected {}".format(name, target.shape, sample.shape)
                )
            target[...] = sample
        if hasattr(data, "time"):
            data.time = float(self.arrays["time"][index])
        if forward:
            if hasattr(physics, "forward"):
                physics.forward()
            else:
                try:
                    import mujoco
                except ImportError as exc:
                    raise RuntimeError("mujoco is required to forward raw MjData") from exc
                mujoco.mj_forward(_model_from_physics(physics), data)
        return float(self.arrays["time"][index])
