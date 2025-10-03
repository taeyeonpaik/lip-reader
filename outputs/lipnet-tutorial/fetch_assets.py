"""Download the two public archives linked by the tutorial author."""

import argparse
import json
import zipfile
from pathlib import Path

import gdown


def unpack(archive_path, target):
    target = Path(target).resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for item in archive.infolist():
            if not (target / item.filename).resolve().is_relative_to(target):
                raise ValueError(f"Unsafe archive path: {item.filename}")
        archive.extractall(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="assets")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    archives = [("1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL", "data.zip", "dataset"),
                ("1vWscXs4Vt0a_1IH1-ct2TCgXAZT-N3_Y", "checkpoints.zip", "pretrained")]
    for identity, filename, folder in archives:
        destination = output / filename
        if not destination.exists():
            result = gdown.download(id=identity, output=str(destination), quiet=False)
            if result is None:
                raise RuntimeError(f"Download failed: {filename}")
        unpack(destination, output / folder)
    print(json.dumps({"data": str(output / "dataset" / "data"),
                      "checkpoint": str(output / "pretrained" / "checkpoint")}, indent=2))


if __name__ == "__main__":
    main()
