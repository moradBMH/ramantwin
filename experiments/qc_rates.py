"""Detection and false-positive rates of the physics QC checks on twin spectra.

Run: `uv run python experiments/qc_rates.py`
"""

import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ramantwin.biochem import sample_cohort
from ramantwin.instrument import InstrumentParams, VirtualInstrument, nominal, sample_fleet
from ramantwin.ml.acquire import render
from ramantwin.ml.qc import physics_checks
from ramantwin.pipeline import calibrate, process

LOW_SNR = {"signal_e_per_s": 500.0, "exposure_s": 0.5, "n_accumulations": 1}


def configs(n_fleet: int = 10) -> dict[str, list[InstrumentParams]]:
    fleet = sample_fleet(n_fleet, np.random.default_rng(11), spread=1.0)
    out = {"nominal": [nominal()], "fleet": fleet}
    out["fleet_low_snr"] = [InstrumentParams.from_dict({**p.to_dict(), **LOW_SNR}) for p in fleet]
    for a in (0.1, 0.3, 0.5, 1.0):
        out[f"ambient_{a}"] = [
            InstrumentParams.from_dict({**p.to_dict(), "ambient_light": a})
            for p in [nominal(), *fleet]
        ]
    return out


def measure(job):
    params, n, seed = job
    rng = np.random.default_rng(seed)
    inst = VirtualInstrument(params)
    calib = calibrate(inst, rng)
    cohort = sample_cohort(n, 1, rng)
    return [
        process(inst.acquire_pixels(s, rng), calib) for s in render(inst, cohort.concentrations)
    ]


def collect(n_per_instrument: int = 20, cache: Path | None = None) -> dict:
    if cache and cache.exists():
        return pickle.loads(cache.read_bytes())
    jobs = {
        k: [(p, n_per_instrument, 100 + i) for i, p in enumerate(v)] for k, v in configs().items()
    }
    with ProcessPoolExecutor() as pool:
        data = {
            k: (v, [proc for batch in pool.map(measure, js) for proc in batch])
            for k, js in jobs.items()
            for v in [[p for p, _, _ in js]]
        }
    if cache:
        cache.write_bytes(pickle.dumps(data))
    return data


def rates(data: dict, **kw) -> None:
    for key, (instruments, procs) in data.items():
        n_inst = len(instruments)
        per = len(procs) // n_inst
        line = snr = 0
        for i, proc in enumerate(procs):
            reasons = [
                r for _, r in physics_checks(proc, instruments[i // per].resolution_cm1, **kw)
            ]
            line += any("room-light" in r for r in reasons)
            snr += any("signal-to-noise" in r for r in reasons)
        n = len(procs)
        print(f"{key:15s} n={n:4d} room-light {line / n:6.1%}  low-SNR {snr / n:6.1%}")


if __name__ == "__main__":
    cache = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    rates(collect(cache=cache))
