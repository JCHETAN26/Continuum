from __future__ import annotations

import argparse
from pathlib import Path

from continuum_trainer.peft_engine import export_onnx_with_optimum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a PEFT-merged model to ONNX.")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--task", default="feature-extraction")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_onnx_with_optimum(args.model_dir, args.output_dir, args.task)


if __name__ == "__main__":
    main()
