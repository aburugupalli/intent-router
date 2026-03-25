from __future__ import annotations

import argparse
import subprocess
import sys


def run(cmd: list[str]) -> int:
    return subprocess.call(cmd)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Intent Router CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("seed")
    sub.add_parser("train")

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("text", type=str)
    p_pred.add_argument("--json", action="store_true")
    p_pred.add_argument("--threshold", type=float, default=0.55)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.cmd == "seed":
        raise SystemExit(run([sys.executable, "src/seed_data.py"]))
    if args.cmd == "train":
        raise SystemExit(run([sys.executable, "src/train.py"]))
    if args.cmd == "predict":
        cmd = [sys.executable, "src/predict.py", args.text, "--threshold", str(args.threshold)]
        if args.json:
            cmd.append("--json")
        raise SystemExit(run(cmd))

    raise SystemExit(2)


if __name__ == "__main__":
    main()
