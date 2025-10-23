import argparse
import json
import tempfile
from pathlib import Path
from .pipeline import LipReader, prepare_video, subtitles

def main():
    parser = argparse.ArgumentParser(description="Transcribe a local video using visual information only")
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="lipreading-") as folder:
        video = Path(folder) / "silent.mp4"
        prepare_video(args.video, video)
        result = LipReader(device=args.device).transcribe(video)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"result.json").write_text(json.dumps(result, indent=2)+"\n")
    (args.output/"transcript.txt").write_text(result["text"]+"\n")
    (args.output/"subtitles.srt").write_text(subtitles(result))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
