"""Preserve tracked and new Python sources used by a benchmark invocation."""
import hashlib
import json
from pathlib import Path
import shutil


def snapshot(root, outdir):
    root, outdir = Path(root), Path(outdir)
    files = [root / 'main.py']
    for directory in ('MoEModule', 'utils', 'gpt_output'):
        files.extend((root / directory).rglob('*.py'))
        files.extend((root / directory).rglob('*.cpp'))
    hashes = {}
    for source in sorted(set(files)):
        relative = source.relative_to(root)
        target = outdir / 'source_files' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    (outdir / 'source_sha256.json').write_text(json.dumps(hashes, indent=2))
