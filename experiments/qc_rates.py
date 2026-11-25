"""Detection and false-positive rates of the physics QC checks on twin spectra.

Sweeps the instrument resolution (4-20 cm-1) at high and low signal-to-noise ratio, with and
without room light, on fleet instruments drawn with `seed`.
Run: `uv run python experiments/qc_rates.py [seed]` (thresholds were tuned with seed 11 and are
reported on other seeds).
"""

import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ramantwin.biochem import sample_cohort
from ramantwin.instrument import InstrumentParams, VirtualInstrument, sample_fleet
from ramantwin.ml.acquire import render
from ramantwin.ml.qc import physics_checks
from ramantwin.pipeline import calibrate, process

RESOLUTIONS = (4.0, 6.0, 10.0, 14.0, 20.0)
SNR = {
    "high": {"n_accumulations": 30},
    "low": {"signal_e_per_s": 500.0, "exposure_s": 0.5, "n_accumulations": 1},
}
AMBIENT = (0.0, 0.3, 0.5)


def configs(seed: int, n_instruments: int = 4) -> dict[tuple, list[InstrumentParams]]:
    fleet = sample_fleet(n_instruments, np.random.default_rng(seed), spread=1.0)
    out = {}
    for res in RESOLUTIONS:
        for snr, changes in SNR.items():
            for ambient in AMBIENT:
                extra = {**changes, "resolution_cm1": res, "ambient_light": ambient}
                out[(res, snr, ambient)] = [
                    InstrumentParams.from_dict({**p.to_dict(), **extra}) for p in fleet
                ]
    return out


def measure(job) -> list[list[str]]:
    params, n, seed = job
    rng = np.random.default_rng(seed)
    inst = VirtualInstrument(params)
    calib = calibrate(inst, rng)
    cohort = sample_cohort(n, 1, rng)
    reasons = []
    for s in render(inst, cohort.concentrations):
        proc = process(inst.acquire_pixels(s, rng), calib)
        issues = physics_checks(proc, params.resolution_cm1, laser_nm=params.laser_nm)
        reasons.append([r for _, r in issues])
    return reasons


def rates(seed: int, n_per_instrument: int = 15) -> dict[tuple, tuple[float, float]]:
    """(room-light rate, low-SNR rate) per (resolution, snr, ambient) configuration."""
    jobs = {
        key: [(p, n_per_instrument, 1000 * seed + 10 * i) for i, p in enumerate(ps)]
        for key, ps in configs(seed).items()
    }
    out = {}
    with ProcessPoolExecutor() as pool:
        for key, js in jobs.items():
            reasons = [r for batch in pool.map(measure, js) for r in batch]
            line = np.mean([any("room light" in x for x in r) for r in reasons])
            snr = np.mean([any("signal-to-noise" in x for x in r) for r in reasons])
            out[key] = (float(line), float(snr))
    return out


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    table = rates(seed)
    print(f"seed {seed}: room-light flagged rate, then low-SNR flagged rate at ambient 0")
    print("res cm-1 | high SNR amb 0 / 0.3 / 0.5 | low SNR amb 0 / 0.3 / 0.5")
    for res in RESOLUTIONS:
        cells = []
        for snr in SNR:
            cells.append(" ".join(f"{table[(res, snr, a)][0]:6.1%}" for a in AMBIENT))
        high_snr, low_snr = table[(res, "high", 0.0)][1], table[(res, "low", 0.0)][1]
        print(f"{res:8.0f} | {cells[0]} | {cells[1]} | low-SNR {high_snr:.0%} / {low_snr:.0%}")


if __name__ == "__main__":
    main()
