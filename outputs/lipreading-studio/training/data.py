"""Import pinned public LRS2 mouth-crop parquet files into a local manifest."""
import argparse
import hashlib
import json
from pathlib import Path

from training.metrics import normalize


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def validate_splits(rows):
    identities, hashes = {}, {}
    for row in rows:
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError("Unknown dataset split")
        if not normalize(row["text"]):
            raise ValueError(f"Empty reference for {row['id']}")
        if row["id"] in identities:
            raise ValueError(f"Duplicate sample ID: {row['id']}")
        identities[row["id"]] = row["split"]
        previous = hashes.get(row["sha256"])
        if previous is not None and previous != row["split"]:
            raise ValueError("Identical video occurs across dataset splits")
        hashes[row["sha256"]] = row["split"]


def import_data(downloads, destination, train_limit=4000, validation_limit=500):
    import pyarrow.parquet as pq
    destination = Path(destination).resolve()
    videos = destination / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    rows = []
    sources = json.loads(Path(downloads).read_text())
    for source in sources:
        path = Path(source["local_file"])
        with path.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        if digest != source["sha256"]:
            raise ValueError(f"Downloaded file changed: {path.name}")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            for item in batch.to_pylist():
                identity = item["sample_id"]
                key = hashlib.sha256(identity.encode()).hexdigest()[:24]
                video = item["video"]
                if not isinstance(video, bytes):
                    raise ValueError("Expected embedded video bytes")
                output = videos / (key + ".mp4")
                if not output.exists():
                    output.write_bytes(video)
                rows.append({"id": identity, "split": source["split"],
                             "path": str(output), "text": normalize(item["label"]),
                             "declared_frames": item["length"],
                             "sha256": hashlib.sha256(video).hexdigest(),
                             "source_repository": source["repository"],
                             "source_revision": source["revision"]})
    validate_splits(rows)
    # Hash ordering fixes the sample selection before any model predictions.
    rows.sort(key=lambda row: hashlib.sha256(("studio-v1:" + row["id"]).encode()).hexdigest())
    selected = []
    for split, limit in (("train", train_limit), ("validation", validation_limit), ("test", None)):
        selected.extend([r for r in rows if r["split"] == split][:limit])
    validate_splits(selected)
    groups = {s: {r["id"].split("/")[1] for r in selected if r["split"] == s}
              for s in ("train", "validation", "test")}
    report = {
        "dataset": "LRS2, public preprocessed mouth-crop distribution by MahmoodAnaam",
        "input_format": "25 fps, 96×96 RGB mouth crops; not full-face camera videos",
        "splits": {s: sum(r["split"] == s for r in selected) for s in groups},
        "download_sources": [{k: v for k, v in s.items() if k != "local_file"} for s in sources],
        "selection": "Downloaded training shards listed in download_sources, then deterministic SHA256 sample ordering; capped train and validation; all clean test rows",
        "duplicate_ids": 0, "identical_video_across_splits": 0,
        "recording_id_overlap": {f"{a}/{b}": len(groups[a] & groups[b]) for a,b in (("train","validation"),("train","test"),("validation","test"))},
        "limitations": ["Retains distributor's original split assignments, not a new speaker-independent split",
                       "Base model already trained on LRS2; adaptation validation cannot establish unseen-speaker generalization",
                       "This evaluates prepared crops; face tracking and arbitrary uploads require separate tests",
                       "Distributor preprocessing details and original media rights are not fully documented; raw dataset is excluded from the application package"]}
    atomic_json(destination / "manifest.json", selected)
    atomic_json(destination / "provenance.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=4000)
    parser.add_argument("--validation-limit", type=int, default=500)
    args = parser.parse_args()
    print(json.dumps(import_data(args.downloads, args.destination, args.train_limit, args.validation_limit), indent=2))
