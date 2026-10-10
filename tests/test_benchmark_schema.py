"""Round-trip of the v0 syndrome-dataset format.

Uses a Stim-generated repetition code with generic depolarizing noise purely as a FORMAT
test -- this is not one of our physical noise models.
"""
import numpy as np
import pytest

from bosonoise.benchmark import read_dataset, write_dataset


def test_roundtrip(tmp_path):
    stim = pytest.importorskip("stim")
    c = stim.Circuit.generated("repetition_code:memory", distance=3, rounds=3, after_clifford_depolarization=0.01)
    det, obs = c.compile_detector_sampler(seed=1).sample(200, separate_observables=True)
    rng = np.random.default_rng(2)
    heralds = rng.random((200, 5)) < 0.1
    soft = rng.normal(size=(200, 4)).astype(np.float32)
    meta = dict(code={"family": "repetition", "distance": 3, "rounds": 3, "basis": "Z"},
                noise_model={"name": "format-test-depolarizing", "physical": False},
                generator={"seed": 1}, layout={"heralds": "5 dummy herald bits", "soft": "4 dummy floats"})
    path = write_dataset(tmp_path / "ds", c, det, obs, metadata=meta, heralds=heralds, soft=soft)
    back = read_dataset(path)
    np.testing.assert_array_equal(back["detectors"], det)
    np.testing.assert_array_equal(back["observables"], obs)
    np.testing.assert_array_equal(back["heralds"], heralds)
    np.testing.assert_allclose(back["soft"], soft)
    assert back["circuit"] == c
    assert back["metadata"]["schema_version"] == "0.1"
    assert back["metadata"]["shots"] == 200


def test_b8_matches_stim_and_survives_non_ascii_paths(tmp_path):
    """Our numpy b8 writer/reader equals Stim's own b8 (ASCII temp path) and works under a non-ASCII directory."""
    stim = pytest.importorskip("stim")
    from bosonoise.benchmark.schema import read_b8, write_b8

    rng = np.random.default_rng(3)
    for width in (1, 7, 8, 13, 90):
        arr = rng.random((37, width)) < 0.3
        ours = tmp_path / f"ours_{width}.b8"
        write_b8(ours, arr)
        ref = tmp_path / f"stim_{width}.b8"
        stim.write_shot_data_file(data=arr, path=str(ref), format="b8", num_measurements=width)
        assert ours.read_bytes() == ref.read_bytes()
        assert np.array_equal(read_b8(ours, width), arr)
    uni = tmp_path / "Рабочий стол"
    uni.mkdir()
    write_b8(uni / "x.b8", arr)
    assert np.array_equal(read_b8(uni / "x.b8", arr.shape[1]), arr)
