"""Sanity check without the twin: does augmentation protect the CNN against band shifts?

Trains the SVM, a plain CNN and an augmented CNN on unshifted synthetic spectra, then tests them
on new patients measured with growing x-axis shifts.
Run: `uv run python experiments/synthetic_shift.py`
"""

from ramantwin.ml.augment import AugmentParams
from ramantwin.ml.evaluate import binary_metrics
from ramantwin.ml.models import cnn_outputs
from ramantwin.ml.synthetic import make_spectra
from ramantwin.ml.train import resolve_device, train_cnn, train_svm

SHIFTS = (0.0, 4.0, 8.0, 12.0)


def main() -> None:
    train = make_spectra(n_patients=100, spectra_per_patient=4, n_instruments=2, seed=0)
    tests = {s: make_spectra(60, 3, 1, seed=1, shifts_cm1=(s,)) for s in SHIFTS}
    device = resolve_device()
    models = {
        "svm": train_svm(train.X, train.y),
        "cnn": train_cnn(train.X, train.y, train.patient, epochs=20, device=device).model,
        "cnn+aug": train_cnn(
            train.X, train.y, train.patient, augment=AugmentParams(), epochs=20, device=device
        ).model,
    }
    print("balanced accuracy by shift (cm-1):", SHIFTS)
    for name, model in models.items():
        row = []
        for d in tests.values():
            p = model.predict_proba(d.X)[:, 1] if name == "svm" else cnn_outputs(model, d.X)[0]
            row.append(binary_metrics(p, d.y)["bacc"])
        print(f"{name:8s}", " ".join(f"{v:.3f}" for v in row))


if __name__ == "__main__":
    main()
