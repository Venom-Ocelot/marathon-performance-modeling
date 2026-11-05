"""Download only the pinned 2019 daily running-log file (about 956 MB extracted)."""
from kaggle_download import ROOT, extract, fetch_archive, write_manifest

REF = 'mexwell/long-distance-running-dataset'
FILENAME = 'run_ww_2019_d.csv'


def main():
    folder = ROOT / 'data/raw/training_logs'
    archive = fetch_archive(REF, 1, folder / 'source_download', filename=FILENAME)
    files = extract(archive, folder, [FILENAME])
    write_manifest(folder / 'manifest.json', folder, files, REF, 1,
                   'CC BY 4.0', selected_file=FILENAME)
    print('2019 daily logs ready; source hash verified.')


if __name__ == '__main__':
    main()
