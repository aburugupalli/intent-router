from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def test_train_and_predict(tmp_path: Path) -> None:
    data_out = tmp_path / "train.csv"
    model_out = tmp_path / "intent_router.joblib"

    run([sys.executable, "src/seed_data.py", "--out", str(data_out)])
    run([sys.executable, "src/train.py", "--data", str(data_out), "--model-out", str(model_out)])

    # Basic predict should work
    result = subprocess.run(
        [
            sys.executable,
            "src/predict.py",
            "Ich wurde doppelt abgebucht",
            "--model",
            str(model_out),
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"intent"' in result.stdout
