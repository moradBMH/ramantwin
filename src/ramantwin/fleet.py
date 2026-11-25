"""Fleet datasets: cohorts of patients measured on many virtual instruments.

CLI: python -m ramantwin.fleet --preset demo --out data/demo
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ramantwin.biochem import COMPONENT_NAMES, Cohort, component_matrix, sample_cohort
from ramantwin.constants import CLASSES, GRID, GRID_START, GRID_STEP, GRID_STOP
from ramantwin.instrument import InstrumentParams, VirtualInstrument, sample_fleet
from ramantwin.pipeline import calibrate, process, raw_on_grid

TRAIN_SPREAD = 0.15
FLEET_SPREAD = 1.0

PRESETS: dict[str, dict] = {
    "demo": {"n_train": 2, "n_fleet": 10, "patients": 60, "spectra": 5, "seed": 2026},
    "tiny": {"n_train": 1, "n_fleet": 2, "patients": 6, "spectra": 2, "seed": 7},
}


@dataclass
class FleetDataset:
    X_raw: np.ndarray
    X: np.ndarray
    y: np.ndarray
    patient: np.ndarray
    instrument: np.ndarray
    concentrations: np.ndarray
    instruments: list[InstrumentParams]
    roles: list[str]

    def save(self, path: Path) -> None:
        """Write `path`.npz (arrays) and `path`.json (instruments, roles, metadata)."""
        path = _stem(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            f"{path}.npz",
            X_raw=self.X_raw,
            X=self.X,
            y=self.y,
            patient=self.patient,
            instrument=self.instrument,
            concentrations=self.concentrations,
        )
        meta = {
            "instruments": [p.to_dict() for p in self.instruments],
            "roles": self.roles,
            "classes": list(CLASSES),
            "component_names": list(COMPONENT_NAMES),
            "grid": {"start": GRID_START, "stop": GRID_STOP, "step": GRID_STEP},
        }
        Path(f"{path}.json").write_text(json.dumps(meta, indent=1))

    @classmethod
    def load(cls, path: Path) -> "FleetDataset":
        path = _stem(path)
        arrays = np.load(f"{path}.npz")
        meta = json.loads(Path(f"{path}.json").read_text())
        return cls(
            **{k: arrays[k] for k in arrays.files},
            instruments=[InstrumentParams.from_dict(d) for d in meta["instruments"]],
            roles=list(meta["roles"]),
        )

    def subset(self, mask: np.ndarray) -> "FleetDataset":
        """Select spectra; `instruments` and `roles` are kept so indices stay valid."""
        return FleetDataset(
            X_raw=self.X_raw[mask],
            X=self.X[mask],
            y=self.y[mask],
            patient=self.patient[mask],
            instrument=self.instrument[mask],
            concentrations=self.concentrations[mask],
            instruments=self.instruments,
            roles=self.roles,
        )


def _stem(path: Path) -> Path:
    path = Path(path)
    return path.with_suffix("") if path.suffix in (".npz", ".json") else path


def acquire_cohort(
    cohort: Cohort, params: InstrumentParams, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Calibrate one instrument, measure every spectrum of `cohort`, return (X_raw, X)."""
    rng = np.random.default_rng(seed)
    instrument = VirtualInstrument(params)
    calib = calibrate(instrument, rng)
    # line shape and pixel sampling are linear, so mixing pre-rendered components is exact
    components = instrument.line_shape(component_matrix)

    n = len(cohort.labels)
    X_raw = np.empty((n, GRID.size), dtype=np.float32)
    X = np.empty((n, GRID.size), dtype=np.float32)
    for i, c in enumerate(cohort.concentrations):
        acq = instrument.acquire_pixels(c @ components, rng)
        X_raw[i] = raw_on_grid(acq, instrument)
        X[i] = process(acq, calib).final
    return X_raw, X


def _seed(seed: int, patient_offset: int, index: int, stream: int) -> int:
    entropy = [seed, patient_offset, index, stream]
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])


def _measure(job: tuple) -> tuple[Cohort, np.ndarray, np.ndarray]:
    index, params, patients, spectra, seed, offset = job
    cohort = sample_cohort(
        patients,
        spectra,
        np.random.default_rng(_seed(seed, offset, index, 0)),
        patient_offset=offset + index * patients,
    )
    X_raw, X = acquire_cohort(cohort, params, _seed(seed, offset, index, 1))
    return cohort, X_raw, X


def make_dataset(
    instruments: list[InstrumentParams],
    roles: list[str],
    patients_per_instrument: int,
    spectra_per_patient: int,
    seed: int,
    n_jobs: int = -1,
    patient_offset: int = 0,
) -> FleetDataset:
    """Each instrument measures its own patients; one process per instrument when n_jobs != 1.

    Patient ids start at `patient_offset` and patients are drawn from (`seed`, `patient_offset`),
    so two calls with the same seed and offset return the same patients. Give every dataset that
    must not share patients with another one its own offset (or seed).
    """
    jobs = [
        (i, params, patients_per_instrument, spectra_per_patient, seed, patient_offset)
        for i, params in enumerate(instruments)
    ]
    workers = (os.cpu_count() or 1) if n_jobs == -1 else n_jobs
    if workers == 1:
        results = [_measure(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            results = list(pool.map(_measure, jobs))

    cohorts = [r[0] for r in results]
    return FleetDataset(
        X_raw=np.concatenate([r[1] for r in results]),
        X=np.concatenate([r[2] for r in results]),
        y=np.concatenate([c.labels for c in cohorts]).astype(np.int8),
        patient=np.concatenate([c.patient_ids for c in cohorts]).astype(np.int32),
        instrument=np.concatenate(
            [np.full(len(c.labels), i, dtype=np.int16) for i, c in enumerate(cohorts)]
        ),
        concentrations=np.concatenate([c.concentrations for c in cohorts]).astype(np.float32),
        instruments=list(instruments),
        roles=list(roles),
    )


def preset_instruments(name: str) -> tuple[list[InstrumentParams], list[str]]:
    """Montréal train instruments (small spread) followed by the fleet (full spread)."""
    cfg = PRESETS[name]
    rng = np.random.default_rng(cfg["seed"])
    train = sample_fleet(cfg["n_train"], rng, spread=TRAIN_SPREAD, prefix="MTL")
    fleet = sample_fleet(cfg["n_fleet"], rng, spread=FLEET_SPREAD)
    return train + fleet, ["train"] * len(train) + ["fleet"] * len(fleet)


def make_preset(name: str, n_jobs: int = -1) -> FleetDataset:
    cfg = PRESETS[name]
    instruments, roles = preset_instruments(name)
    return make_dataset(instruments, roles, cfg["patients"], cfg["spectra"], cfg["seed"], n_jobs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a simulated multi-instrument dataset.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="demo")
    parser.add_argument("--out", type=Path, default=Path("data/demo"))
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    start = time.perf_counter()
    ds = make_preset(args.preset, args.n_jobs)
    ds.save(args.out)
    elapsed = time.perf_counter() - start
    print(f"{len(ds.y)} spectra, {len(ds.instruments)} instruments, {elapsed:.1f} s")
    print(f"wrote {_stem(args.out)}.npz and {_stem(args.out)}.json")


if __name__ == "__main__":
    main()
