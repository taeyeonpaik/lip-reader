"""Restore pinned, ignored third-party sources; never download datasets or weights."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
LOCK = Path(__file__).with_name('dependencies.json')


def fetch(url):
    with urlopen(url, timeout=120) as response:
        return response.read()


def install(name, spec):
    destination = ROOT / spec['destination']
    if destination.exists() and all((destination / f).is_file() for f in spec['files']):
        verify(destination, spec['files'])
        print(f'{name}: already verified')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='lip-reader-dependency-') as temp:
        staging = Path(temp) / 'source'
        staging.mkdir()
        if 'repository' in spec:
            url = f"https://github.com/{spec['repository']}/archive/{spec['revision']}.zip"
            with ZipFile(io.BytesIO(fetch(url))) as archive:
                for member in archive.infolist():
                    relative = Path(*Path(member.filename).parts[1:])
                    if member.is_dir() or str(relative) == '.':
                        continue
                    target = staging / relative
                    if not target.resolve().is_relative_to(staging.resolve()):
                        raise ValueError('Unsafe dependency archive path')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(member))
        for relative, url in spec.get('downloads', {}).items():
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(fetch(url))
        for relative in spec.get('empty_files', []):
            (staging / relative).touch()
        if spec.get('patch'):
            subprocess.run(['git', 'apply', str(ROOT / spec['patch'])],
                           cwd=staging, check=True)
        verify(staging, spec['files'])
        if destination.exists():
            if 'repository' in spec:
                raise RuntimeError(f'Refusing to overwrite incomplete dependency: {destination}')
            for relative in spec['files']:
                target = destination / relative
                if target.exists():
                    verify(destination, {relative: spec['files'][relative]})
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(staging / relative, target)
        else:
            shutil.copytree(staging, destination)
    print(f'{name}: installed and verified')


def verify(directory, files):
    for relative, expected in files.items():
        path = directory / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Dependency checksum mismatch: {path}')


def main():
    specs = json.loads(LOCK.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('names', nargs='*', default=['autoavsr'])
    args = parser.parse_args()
    for name in args.names:
        if name not in specs:
            parser.error('Unknown dependency: ' + name)
        install(name, specs[name])


if __name__ == '__main__':
    main()
