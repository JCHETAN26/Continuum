"""Materialise the embedding model into a plain directory.

Container images run this at build time so nothing reaches the network at startup, and
every replica serves byte-identical weights. Drift compares centroids across time, so a
replica quietly running different weights would register as drift that never happened.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# Deliberately not imported from continuum_shared. That package instantiates Settings at
# import time, which requires DATABASE_URL and friends — none of which exist during a
# docker build. test_fetch_embedding_model.py asserts these stay equal to the module's.
MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"
ONNX_FILENAME = "onnx/model.onnx"
TOKENIZER_FILENAME = "tokenizer.json"


def fetch(target: Path) -> Path:
    from huggingface_hub import hf_hub_download

    target.mkdir(parents=True, exist_ok=True)
    for filename, local_name in ((ONNX_FILENAME, "model.onnx"), (TOKENIZER_FILENAME, None)):
        source = Path(hf_hub_download(MODEL_REPO, filename))
        shutil.copyfile(source, target / (local_name or source.name))
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()

    destination = fetch(args.target)
    for path in sorted(destination.iterdir()):
        print(f"{path.name}\t{path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
