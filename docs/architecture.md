# Architecture and interface contracts

RamanTwin answers one question: **is a Raman AI model validated on one instrument safe on the next one?**
It simulates a fleet of Raman instruments (the *twin*), processes every acquisition with the
real ORPL pipeline, trains models, measures how they degrade across instruments, hardens them,
and shows the result in an interactive 3D web app.

```
biochem ──► instrument (twin) ──► pipeline (ORPL) ──► fleet dataset ──► ml (train / qc / evaluate)
                                                                              │
                                                     web/public/data/*.json ◄─┴─► api (FastAPI) ◄── web (three.js)
```

The contracts below are **frozen**. Change them only through a dedicated PR that updates this file first.

## 1. Shared constants (`ramantwin.constants`)

| Name | Value | Meaning |
|---|---|---|
| `N_PIX` | 1024 | CCD pixels of every virtual instrument |
| `LASER_NM` | 785.0 | nominal excitation wavelength |
| `GRID` | 600 → 1800 cm⁻¹, step 1 (1201 points) | common axis of every processed spectrum |
| `TRUNCATE_BELOW_CM1` | 450 | raw pixels below this shift are discarded before baseline removal |
| `CLASSES` | `("normal", "cancer")` | label 0, label 1 |
| `TYLENOL_CAL_PEAKS_CM1` | 797, 858, 1169, 1237, 1324, 1609, 1648 | peaks used by `orpl.calibration.autogenx(preset="tylenol")` |

`import ramantwin` applies `ramantwin.orpl_compat`, which fixes `orpl.calibration.xaxis_from_peaks`
under NumPy 2.x. Always import `ramantwin` before calling ORPL calibration.

## 2. `ramantwin.biochem` — tissue model

```python
@dataclass(frozen=True)
class Band:
    center: float   # cm-1
    fwhm: float     # cm-1
    height: float   # relative

COMPONENTS: dict[str, tuple[Band, ...]]        # literature band assignments
COMPONENT_NAMES = ("protein", "lipid", "nucleic_acid", "carotenoid", "cholesterol", "collagen", "heme")
                                               # frozen names and order of concentration arrays / API keys

def component_spectrum(name: str, xaxis: np.ndarray) -> np.ndarray          # Lorentzian sum, max = 1
def raman_spectrum(concentrations: np.ndarray, xaxis: np.ndarray) -> np.ndarray
    # concentrations (C,) -> (len(xaxis),)   or (n, C) -> (n, len(xaxis)); noiseless

@dataclass
class Cohort:
    concentrations: np.ndarray   # (n, C) float
    labels: np.ndarray           # (n,) int, 0 normal / 1 cancer
    patient_ids: np.ndarray      # (n,) int, globally unique
    component_names: tuple[str, ...]

def sample_cohort(n_patients: int, spectra_per_patient: int, rng: np.random.Generator,
                  cancer_fraction: float = 0.5, patient_offset: int = 0) -> Cohort
```

Requirements: class differences follow published trends (e.g. lower carotenoids at 1156/1520 cm⁻¹,
higher nucleic acids, protein/lipid ratio shifts), with patient-level random effects and intra-patient
variability so that **in-domain accuracy of a linear SVM is about 88–95 %, not 100 %**.

## 3. `ramantwin.instrument` — the twin

```python
@dataclass
class InstrumentParams:
    name: str = "nominal"
    laser_nm: float = 785.0            # excitation at calibration time
    laser_drift_nm: float = 0.0        # drift after calibration (not corrected by calibration)
    x_offset_cm1: float = 0.0          # spectrometer alignment offset (corrected by calibration)
    x_scale: float = 1.0               # dispersion scale (corrected by calibration)
    resolution_cm1: float = 6.0        # instrument line shape FWHM
    irf_tilt: float = 0.0              # instrument response slope across the sensor
    irf_curvature: float = 0.0
    etalon_amp: float = 0.0            # CCD etaloning ripple
    fluorescence_ratio: float = 5.0    # max(background) / max(raman)
    fluorescence_shape: float = 1.0
    signal_e_per_s: float = 2000.0     # photo-electrons per second at the strongest Raman band
    exposure_s: float = 1.0
    n_accumulations: int = 10
    read_noise_e: float = 8.0
    dark_e_per_s: float = 2.0
    cosmic_rate: float = 0.05          # cosmic rays per frame
    ambient_light: float = 0.0         # room-light emission lines (QC trigger)
    saturation_e: float = 65000.0

    def to_dict(self) -> dict
    @classmethod
    def from_dict(cls, d: dict) -> "InstrumentParams"

SLIDER_RANGES: dict[str, tuple[float, float]]   # physical ranges exposed in the web Twin Lab

def nominal() -> InstrumentParams
def sample_fleet(n: int, rng: np.random.Generator, spread: float = 1.0,
                 prefix: str = "SN") -> list[InstrumentParams]

@dataclass
class Acquisition:
    accumulations: np.ndarray   # (n_acc, N_PIX) counts (gain 1: counts = photo-electrons)
    dark: np.ndarray            # (N_PIX,) mean dark frame, same exposure

class VirtualInstrument:
    def __init__(self, params: InstrumentParams)
    def pixel_shift_cm1(self, drift: bool = True) -> np.ndarray   # true Raman shift of each pixel
    def nominal_axis_cm1(self) -> np.ndarray                      # factory axis (no calibration)
    def acquire(self, spectrum_fn: Callable[[np.ndarray], np.ndarray], rng,
                fluorescence: bool = True) -> Acquisition
        # spectrum_fn maps Raman shift (cm-1) -> noiseless intensity (max ~ 1)
    def acquire_tylenol(self, rng) -> Acquisition                 # used by calibration, no drift
    def acquire_nist(self, rng) -> Acquisition                    # NIST SRM 2241 luminescence, no drift
```

Every instrument covers at least 450–1900 cm⁻¹ on its 1024 pixels.

`acquire_tylenol` has no tissue fluorescence, and within the camera range its 5 most prominent bands
must belong to `TYLENOL_CAL_PEAKS_CM1` (ORPL's `autogenx` fits the 5 most prominent peaks of the whole
spectrum against 5-of-7 combinations of those positions).

## 4. `ramantwin.pipeline` — ORPL processing (same order as the ROWS study)

```python
@dataclass
class Calibration:
    xaxis: np.ndarray        # (N_PIX,) cm-1, from orpl.calibration.autogenx on Tylenol
    irf: np.ndarray          # (N_PIX,), from orpl.calibration.compute_irf on NIST
    x_residual: float        # polyfit residual of the best Tylenol peak combination (cm-1²)

@dataclass
class Processed:
    # All pixel-space arrays are TRUNCATED to the pixels whose calibrated shift is >= 450 cm-1,
    # so they share one length M <= N_PIX with `xaxis`.
    xaxis: np.ndarray        # (M,) calibrated axis, cm-1
    raw_mean: np.ndarray     # (M,) dark-subtracted, cosmic-ray filtered, averaged counts
    corrected: np.ndarray    # (M,) raw_mean / irf
    baseline: np.ndarray     # (M,) BubbleFill baseline of `corrected`
    raman: np.ndarray        # (M,) corrected - baseline
    final: np.ndarray        # (len(GRID),) raman interpolated on GRID, SNV-normalised
    max_count_fraction: float  # max raw count over all frames / saturation_e (QC: saturation)

def calibrate(instrument: VirtualInstrument, rng) -> Calibration
def process(acq: Acquisition, calib: Calibration, min_bubble_width: int = ...) -> Processed
def raw_on_grid(acq: Acquisition, instrument: VirtualInstrument) -> np.ndarray
    # "raw" view: dark-subtracted mean counts on the factory axis, interpolated on GRID,
    # divided by its max. No calibration, no baseline removal.
```

Order: dark subtraction → `orpl.cosmic_ray.crfilter_multi` → mean → calibrated x-axis →
IRF correction → truncate < 450 cm⁻¹ → `orpl.baseline_removal.bubblefill` → interpolate on `GRID` →
`orpl.normalization.snv`.

## 5. `ramantwin.fleet` — datasets

```python
@dataclass
class FleetDataset:
    X_raw: np.ndarray          # (N, len(GRID)) float32, raw_on_grid
    X: np.ndarray              # (N, len(GRID)) float32, ORPL-processed
    y: np.ndarray              # (N,) int8
    patient: np.ndarray        # (N,) int32
    instrument: np.ndarray     # (N,) int16, index into `instruments`
    concentrations: np.ndarray # (N, C) float32
    instruments: list[InstrumentParams]
    roles: list[str]           # per instrument: "train" or "fleet"

    def save(self, path: Path) -> None        # path.npz + path.json (instruments, roles)
    @classmethod
    def load(cls, path: Path) -> "FleetDataset"
    def subset(self, mask: np.ndarray) -> "FleetDataset"

def acquire_cohort(cohort: Cohort, params: InstrumentParams, seed: int) -> tuple[np.ndarray, np.ndarray]
    # returns (X_raw, X) for every spectrum of the cohort measured on one instrument
def make_dataset(instruments: list[InstrumentParams], roles: list[str], patients_per_instrument: int,
                 spectra_per_patient: int, seed: int, n_jobs: int = -1) -> FleetDataset
```

CLI: `python -m ramantwin.fleet --preset demo --out data/demo` builds 2 "train" instruments
(Montréal, small spread) and 10 "fleet" instruments (spread 1.0). Each instrument measures its own patients.

## 6. `ramantwin.ml` — models, QC, evaluation, export

- `models.py`: `SpectralCNN(nn.Module)`, input `(B, len(GRID))`, `forward -> logits (B, 2)`,
  `embed -> (B, D)`; `make_svm()` returns a scikit-learn linear SVM pipeline with probabilities.
- `augment.py`: augmentations in processed space (sub-pixel shift, broadening, noise, residual baseline).
- `train.py`: three models.
  - `svm`: linear SVM on ORPL-processed spectra, the approach of the ROWS study.
  - `naive`: CNN trained on the train instruments only.
  - `robust`: CNN trained on the same patients re-acquired through randomised twins
    (domain randomisation), plus spectral augmentation.
- `qc.py`: `QCResult(flag: "ok"|"warn"|"reject", ood_score: float, threshold: float, reasons: list[str])`;
  Mahalanobis OOD on CNN embeddings and physics checks (saturation, ambient-light lines, low SNR) computed
  from `Processed`. Fleet-level numbers (`fleet.json` `abstain_rate`, `shift_scene.json` `ood`) use the
  OOD score only, with abstention = `reject`; the API applies OOD + physics checks.
- The ml package owns the artifact formats in `artifacts/` (models, QC statistics) and exposes one
  loader used by both `export.py` and `api.py`, so the website numbers and the Twin Lab agree.
- `evaluate.py`: leave-instruments-out metrics (balanced accuracy, AUC, ECE), risk–coverage, robustness grid.
  Splits are always grouped by patient.
- `export.py`: writes the JSON files of section 8 plus `artifacts/robust.onnx`, and benchmarks latency.

Artifacts (`artifacts/`, `data/`) are not committed. `web/public/data/*.json` is committed.

## 7. `ramantwin.api` — FastAPI

| Method | Path | Body / response |
|---|---|---|
| GET | `/api/health` | `{"status": "ok"}` |
| GET | `/api/config` | `{"params": InstrumentParams, "ranges": SLIDER_RANGES, "grid": {"start", "stop", "step"}, "classes": [...], "models": ["svm", "naive", "robust"]}` |
| POST | `/api/simulate` | request `{"params": {partial InstrumentParams}, "label": 0 \| 1, "seed": int}` |

`/api/simulate` response:

```json
{
  "raw": {"x": [...], "y": [...]},
  "baseline": {"x": [...], "y": [...]},
  "processed": {"x": [...], "y": [...]},
  "truth": {"label": 1, "concentrations": {"carotenoid": 0.4}},
  "predictions": {"svm": {"p_cancer": 0.71}, "naive": {"p_cancer": 0.12}, "robust": {"p_cancer": 0.93}},
  "qc": {"flag": "ok", "ood_score": 3.1, "threshold": 6.0, "reasons": []},
  "timing_ms": {"acquire": 1.2, "orpl": 2.5, "inference": 0.8}
}
```

`raw` is `Processed.corrected` and `baseline` is `Processed.baseline`, both divided by `max(corrected)`
and sharing the truncated calibrated axis (`x` in cm⁻¹), downsampled to at most 600 points.
`processed` is `Processed.final` on `GRID` taken every 2 cm⁻¹ (601 points). `truth.concentrations`
keys are `COMPONENT_NAMES`. Calibration is cached per hardware configuration,
so moving `laser_drift_nm` does **not** recalibrate (that is the point of the demo).
`ramantwin-demo` serves the API and `web/dist` on `http://127.0.0.1:8000`.

## 8. Web data files (`web/public/data/`)

`shift_scene.json`

```json
{
  "version": 1,
  "classes": ["normal", "cancer"],
  "instruments": [{"id": 0, "name": "MTL-01", "role": "train"}],
  "stages": [
    {"key": "raw", "title": "Raw counts", "subtitle": "...", "positions": [0.1, -0.3, 0.7],
     "silhouette": {"instrument": 0.61, "class": 0.02}}
  ],
  "points": {"label": [0], "instrument": [0], "patient": [0], "p_naive": [0.1], "p_robust": [0.2], "ood": [1.3]},
  "spectra": {"grid": [600, 604], "index": [0], "raw": [[0.1, 0.2]], "processed": [[0.1, 0.2]]}
}
```

- Stage keys in order: `raw`, `orpl`, `naive`, `robust`. `positions` is a flat `[x0, y0, z0, x1, ...]`
  array in [-1, 1]³ with the same point order in every stage (so points can be animated between stages).
- At most 4000 points; `spectra` holds at most 400 of them on a 4 cm⁻¹ grid. Floats rounded to 3 decimals.

`robustness_atlas.json`

```json
{
  "version": 1,
  "x": {"param": "laser_drift_nm", "label": "Laser drift (nm)", "values": [-1.0, 0.0, 1.0]},
  "y": {"param": "fluorescence_ratio", "label": "Fluorescence / Raman", "values": [1, 10, 50]},
  "metric": "balanced_accuracy",
  "models": {"svm": [[0.9]], "naive": [[0.9]], "robust": [[0.9]]},
  "train_envelope": {"x": [-0.1, 0.1], "y": [2, 8]}
}
```

`models[m][iy][ix]` is the metric at `y.values[iy]`, `x.values[ix]`.

`fleet.json`

```json
{
  "version": 1,
  "instruments": [{"id": 0, "name": "MTL-01", "role": "train", "params": {},
                   "metrics": {"svm": {"bacc": 0.93, "auc": 0.97}, "naive": {}, "robust": {}},
                   "abstain_rate": 0.02}],
  "summary": {"svm": {"mean_bacc": 0.8, "worst_bacc": 0.6}, "naive": {}, "robust": {}},
  "latency_ms": {"orpl": 2.1, "onnx_cnn": 0.3}
}
```

`twinlab_samples.json`: `{"version": 1, "scenarios": [{"key": "nominal", "title": "...", "params": {}, "response": <simulate response>}]}`.
Scenarios: `nominal`, `laser_drift`, `high_fluorescence`, `ambient_light`, `low_snr`. The web Twin Lab uses
them when the API is unreachable (static mode).
