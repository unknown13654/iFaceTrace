"""Copy before analysis; original files are only hashed and copied, never opened as DBs."""
import hashlib
import json
import shutil
import sys
from pathlib import Path


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def copy_input(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    assert source.is_file() and source != destination
    assert not destination.exists(), 'Refuse to overwrite a copy'
    before = sha256(source)
    original_stat = source.stat()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    copied, after = sha256(destination), sha256(source)
    assert before == copied == after, 'Source changed or copy is incomplete'
    assert original_stat.st_size == source.stat().st_size == destination.stat().st_size
    destination.chmod(0o444 | (original_stat.st_mode & 0o111))
    return dict(source=str(source), copy=str(destination), bytes=original_stat.st_size,
                sha256=before, source_unchanged=True)


if __name__ == '__main__':
    source, destination, receipt = map(Path, sys.argv[1:])
    assert not receipt.exists()
    items = sorted(p for p in source.rglob('*') if p.is_file()) if source.is_dir() else [source]
    records = [copy_input(p, destination / p.relative_to(source) if source.is_dir() else destination) for p in items]
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open('x', encoding='utf-8') as stream:
        json.dump(records, stream, indent=2)
    print(json.dumps(dict(files=len(records), bytes=sum(r['bytes'] for r in records), unchanged=True)))
