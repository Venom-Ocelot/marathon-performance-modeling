"""Download pinned complementary Kaggle sources and verify their provenance."""
from concurrent.futures import ThreadPoolExecutor
import json
from kaggle_download import ROOT, extract, fetch_archive, write_manifest

SOURCES = {
    'chicago': ('runningwithrock/chicago-marathon-results', 3),
    'fall_marathons': ('runningwithrock/2010-2019-fall-marathons', 1),
    'boston_2024': ('runningwithrock/2024-boston-marathon-weather-and-splits', 2),
    'boston_2015_2017': ('rojour/boston-results', 4),
    'training_pilot': ('girardi69/marathon-time-predictions', 2),
}


def download(item):
    name, (ref, version) = item
    folder = ROOT / 'data/raw' / name
    archive = fetch_archive(ref, version, folder / 'source.zip')
    files = extract(archive, folder)
    metadata = json.loads((ROOT / 'data/catalog' / (ref.replace('/', '__') + '.json')).read_text())
    write_manifest(folder / 'manifest.json', folder, files, ref, version, metadata['licenseName'])
    return name


def main():
    with ThreadPoolExecutor(max_workers=3) as pool:
        for name in pool.map(download, SOURCES.items()):
            print(f'{name} ready; source hashes verified.')


if __name__ == '__main__':
    main()
