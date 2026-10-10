"""On-disk format (version 0.1 = SCHEMA_VERSION) of syndrome datasets for the future decoder benchmark.

Every simulated syndrome set is stored from day one in this format, so that it can later
be published unchanged (Zenodo / Hugging Face) and read with Stim/sinter tooling.

Layout of one dataset directory::

    <dataset>/
      metadata.json      schema version, code, noise model (+ parameter provenance),
                         generator (package version, git commit, seed), shot counts, bit layouts
      circuit.stim       the Stim circuit defining DETECTOR / OBSERVABLE_INCLUDE semantics
      detectors.b8       detection events      (stim b8, num_detectors bits per shot)
      observables.b8     observable flips      (stim b8, num_observables bits per shot)
      heralds.b8         optional: herald bits (erasure heralds, buffer-emission flags ...)
      leakage.b8         optional: per-qubit, per-round leakage flags
      soft.npy           optional: float32 soft information, shape (shots, n_soft)

Bit layouts of the optional arrays are described in metadata["layout"].
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import subprocess

import numpy as np

SCHEMA_VERSION = "0.1"


def _git_commit(root: pathlib.Path) -> str | None:
    """HEAD of the repository, with '-dirty' if the code (src, examples, tools) differs from it."""
    try:
        run = lambda *a: subprocess.check_output(["git", *a], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
        return run("rev-parse", "HEAD") + ("-dirty" if run("status", "--porcelain", "--", "src", "examples", "tools") else "")
    except Exception:
        return None


def write_b8(path, arr) -> None:
    """Stim's b8 format (bits of a shot packed little-endian, each shot padded to whole bytes), written with numpy so
    that non-ASCII paths work (Stim's own writer cannot open them on Windows)."""
    a = np.asarray(arr, dtype=bool)
    pathlib.Path(path).write_bytes(np.packbits(a, axis=1, bitorder="little").tobytes())


def read_b8(path, width: int) -> np.ndarray:
    raw = np.frombuffer(pathlib.Path(path).read_bytes(), dtype=np.uint8)
    nbytes = (width + 7) // 8
    return np.unpackbits(raw.reshape(-1, nbytes), axis=1, count=width, bitorder="little").astype(bool)


def write_dataset(path, circuit, detectors: np.ndarray, observables: np.ndarray, *, metadata: dict,
                  heralds: np.ndarray | None = None, leakage: np.ndarray | None = None,
                  soft: np.ndarray | None = None) -> pathlib.Path:
    import stim

    from .. import __version__

    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)
    det = np.asarray(detectors, dtype=bool)
    obs = np.asarray(observables, dtype=bool)
    if det.shape[0] != obs.shape[0]:
        raise ValueError("detectors and observables must have the same number of shots")
    if det.shape[1] != circuit.num_detectors or obs.shape[1] != circuit.num_observables:
        raise ValueError("array widths do not match the circuit")
    (path / "circuit.stim").write_text(str(circuit), encoding="utf-8", newline="\n")  # LF on every platform
    write_b8(path / "detectors.b8", det)
    write_b8(path / "observables.b8", obs)
    extra = {}
    for name, arr in (("heralds", heralds), ("leakage", leakage)):
        if arr is not None:
            arr = np.asarray(arr, dtype=bool)
            write_b8(path / f"{name}.b8", arr)
            extra[name] = int(arr.shape[1])
    if soft is not None:
        np.save(path / "soft.npy", np.asarray(soft, dtype=np.float32))
        extra["soft"] = int(np.asarray(soft).shape[1])
    meta = dict(metadata)
    meta.update(
        schema_version=SCHEMA_VERSION,
        created=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        generator={"package": "bosonoise", "version": __version__,
                   "git_commit": _git_commit(pathlib.Path(__file__).resolve().parents[3]),
                   **meta.get("generator", {})},
        shots=int(det.shape[0]),
        num_detectors=int(det.shape[1]),
        num_observables=int(obs.shape[1]),
        optional_arrays=extra,
    )
    (path / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=False), encoding="utf-8", newline="\n")
    return path


def read_dataset(path) -> dict:
    import stim

    path = pathlib.Path(path)
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    out = dict(metadata=meta, circuit=stim.Circuit((path / "circuit.stim").read_text(encoding="utf-8")))
    out["detectors"] = read_b8(path / "detectors.b8", meta["num_detectors"])
    out["observables"] = read_b8(path / "observables.b8", meta["num_observables"])
    for name, width in meta.get("optional_arrays", {}).items():
        if name == "soft":
            out["soft"] = np.load(path / "soft.npy")
        else:
            out[name] = read_b8(path / f"{name}.b8", width)
    return out
