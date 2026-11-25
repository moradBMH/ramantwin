# RamanTwin

**Is a Raman AI model validated on one instrument safe on the next one?**

RamanTwin is a digital twin of Raman spectrometers. It simulates a fleet of instruments, processes every
acquisition with the open-source [ORPL](https://github.com/mr-sheg/orpl) pipeline (cosmic-ray filtering,
Tylenol x-axis calibration, NIST intensity correction, BubbleFill baseline removal, SNV), and uses it to
stress-test, harden and monitor spectral classifiers across instruments and sites.

> Work in progress. All spectra are simulated; no patient data is used.

## Quick start
```bash
uv sync
uv run pytest
```

See [docs/architecture.md](docs/architecture.md) for the design and interface contracts.

## Acknowledgements
ORPL: G. Sheehy et al., *Open-sourced Raman spectroscopy data processing package implementing a baseline
removal algorithm validated from multiple datasets acquired in human tissue and biofluids*,
J. Biomed. Opt. 28(2), 025002 (2023).
