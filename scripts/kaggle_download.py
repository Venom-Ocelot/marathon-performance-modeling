"""Shared Kaggle authentication, atomic downloads, and provenance handling."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

import requests

ROOT = Path(__file__).resolve().parents[1]


def session():
    """Read local credentials without logging or persisting their values."""
    client = requests.Session()
    config = Path(os.environ.get('KAGGLE_CONFIG_DIR', str(Path.home() / '.kaggle')))
    token = os.environ.get('KAGGLE_API_TOKEN')
    if not token and (config / 'access_token').is_file():
        token = (config / 'access_token').read_text().strip()
    if token:
        client.headers['Authorization'] = 'Bearer ' + token
        return client
    username, key = os.environ.get('KAGGLE_USERNAME'), os.environ.get('KAGGLE_KEY')
    if not (username and key) and (config / 'kaggle.json').is_file():
        legacy = json.loads((config / 'kaggle.json').read_text())
        username, key = legacy.get('username'), legacy.get('key')
    if username and key:
        client.auth = (username, key)
        return client
    raise RuntimeError('Set KAGGLE_API_TOKEN or configure credentials in ~/.kaggle/.')


def fetch_archive(ref, version, destination, filename=None):
    """Cache complete ZIPs; an interrupted transfer never becomes a valid cache."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not zipfile.is_zipfile(destination):
            raise ValueError(f'Invalid cached archive: {destination}. Remove it and retry.')
        return destination
    endpoint = f'https://www.kaggle.com/api/v1/datasets/download/{ref}'
    if filename:
        endpoint += '/' + filename
    partial = destination.with_name(destination.name + '.part')
    try:
        with session() as client:
            with client.get(endpoint, params={'datasetVersionNumber': version},
                            stream=True, timeout=(30, 180)) as response:
                if not response.ok:
                    raise RuntimeError(f'Kaggle returned HTTP {response.status_code} for {ref}.')
                with partial.open('wb') as target:
                    for block in response.iter_content(1024 * 1024):
                        target.write(block)
        if not zipfile.is_zipfile(partial):
            raise ValueError(f'Kaggle did not return a ZIP archive for {ref}.')
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def extract(archive, folder, filenames=None):
    """Stream source files to disk and prevent paths from escaping the destination."""
    folder.mkdir(parents=True, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(archive) as source:
        names = filenames if filenames is not None else [f.filename for f in source.infolist() if not f.is_dir()]
        for name in names:
            target = (folder / name).resolve()
            if not target.is_relative_to(folder.resolve()):
                raise ValueError('Archive member escapes its destination.')
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + '.part')
            try:
                with source.open(name) as src, partial.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
                partial.replace(target)
            finally:
                partial.unlink(missing_ok=True)
            extracted.append(target)
    return extracted


def write_manifest(path, folder, files, ref, version, license_name, selected_file=None):
    hashes = {}
    for file in files:
        with file.open('rb') as handle:
            hashes[str(file.relative_to(folder.resolve()))] = hashlib.file_digest(handle, 'sha256').hexdigest()
    manifest = {'dataset': ref, 'version': version,
                'url': 'https://www.kaggle.com/datasets/' + ref,
                'license_as_listed': license_name,
                'downloaded_utc': datetime.now(timezone.utc).isoformat(), 'files': hashes}
    if selected_file:
        manifest['selected_file'] = selected_file
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get('files') != hashes or previous.get('version') != version:
            raise ValueError(f'Input differs from recorded provenance in {path}; inspect before replacing it.')
        # Preserve original provenance when rebuilding the same cached inputs.
        return
    path.write_text(json.dumps(manifest, indent=2) + '\n')
