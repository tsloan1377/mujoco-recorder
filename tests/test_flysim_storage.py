import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from flysim_storage import ModelMismatchError, TrajectoryRecorder, TrajectoryRun


class FakeModel:
    nq = 3
    nv = 2
    na = 1
    nu = 2
    nbody = 2
    njnt = 1
    nmocap = 0
    nuserdata = 0
    qpos0 = np.array([0.0, 0.0, 1.0])
    body_parentid = np.array([0, 0])
    jnt_type = np.array([3])
    jnt_bodyid = np.array([1])
    jnt_qposadr = np.array([0])
    jnt_dofadr = np.array([0])
    jnt_axis = np.array([[0.0, 0.0, 1.0]])
    jnt_pos = np.zeros((1, 3))

    def id2name(self, index, kind):
        return {
            "body": ["world", "fly"],
            "joint": ["fly/hinge"],
            "actuator": ["fly/motor_a", "fly/motor_b"],
        }[kind][index]


class FakeData:
    def __init__(self):
        self.time = 0.0
        self.qpos = np.zeros(3)
        self.qvel = np.zeros(2)
        self.act = np.zeros(1)
        self.ctrl = np.zeros(2)
        self.mocap_pos = np.zeros((0, 3))
        self.mocap_quat = np.zeros((0, 4))
        self.userdata = np.zeros(0)


class FakePhysics:
    def __init__(self):
        self.model = FakeModel()
        self.data = FakeData()
        self.forward_count = 0

    def forward(self):
        self.forward_count += 1


class StorageTests(unittest.TestCase):
    def test_round_trip_and_replay(self):
        physics = FakePhysics()
        recorder = TrajectoryRecorder(physics, metadata={"seed": 7, "controller": "test"})
        for index in range(4):
            physics.data.time = index * 0.002
            physics.data.qpos[:] = [index, index + 1, index + 2]
            physics.data.qvel[:] = [index * 2, index * 3]
            physics.data.ctrl[:] = [0.1 * index, -0.1 * index]
            recorder.append()

        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp) / "walk.simrun"
            recorder.save(bundle)
            run = TrajectoryRun.load(bundle)
            self.assertEqual(run.sample_count, 4)
            self.assertEqual(run.manifest["provenance"]["seed"], 7)
            self.assertTrue(run.validate_model(physics))

            physics.data.qpos[:] = -1
            applied_time = run.apply(physics, 2)
            np.testing.assert_array_equal(physics.data.qpos, [2, 3, 4])
            self.assertAlmostEqual(applied_time, 0.004)
            self.assertEqual(physics.forward_count, 1)

            manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn("data_sha256", manifest)

    def test_frame_indices_use_nearest_exact_samples(self):
        physics = FakePhysics()
        recorder = TrajectoryRecorder(physics)
        for index in range(11):
            physics.data.time = index * 0.01
            recorder.append()
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp) / "short.simrun"
            recorder.save(bundle)
            run = TrajectoryRun.load(bundle)
            np.testing.assert_array_equal(run.frame_indices(20), [0, 5])

    def test_model_mismatch_is_rejected(self):
        physics = FakePhysics()
        recorder = TrajectoryRecorder(physics)
        recorder.append()
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp) / "short.simrun"
            recorder.save(bundle)
            run = TrajectoryRun.load(bundle)
            physics.model.qpos0 = np.array([9.0, 9.0, 9.0])
            with self.assertRaises(ModelMismatchError):
                run.validate_model(physics)


if __name__ == "__main__":
    unittest.main()
