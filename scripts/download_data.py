"""Download the pinned Berlin source and preserve its SHA-256 manifest."""
from kaggle_download import ROOT, extract, fetch_archive, write_manifest

REF = 'aiaiaidavid/berlin-marathons-data'
VERSION = 2


def main():
    raw = ROOT / 'data/raw'
    archive = fetch_archive(REF, VERSION, raw / 'berlin-marathons-data.zip')
    folder = raw / 'berlin'
    files = extract(archive, folder, ['Berlin_Marathon_data_1974_2019.csv',
                                      'Berlin_Marathon_weather_data_since_1974.csv'])
    write_manifest(raw / 'manifest.json', folder, files, REF, VERSION, 'CC0: Public Domain')
    print('Berlin version 2 ready; source hashes verified.')


if __name__ == '__main__':
    main()
