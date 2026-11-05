"""Check published artifacts without Kaggle credentials or raw input downloads."""
import ast
import base64
import csv
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import re
import subprocess

import nbformat
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


class Report(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images, self.ids, self.anchors = [], set(), []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        if tag == 'img':
            assert attrs.get('alt'), 'Report image needs alternative text.'
            self.images.append(attrs['src'])
        if tag == 'a' and attrs.get('href', '').startswith('#'):
            self.anchors.append(attrs['href'][1:])


def main():
    for path in (ROOT / 'scripts').glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    notebook = nbformat.read(ROOT / 'marathon_analysis.ipynb', as_version=4)
    nbformat.validate(notebook)
    code = [cell for cell in notebook.cells if cell.cell_type == 'code']
    assert [cell.execution_count for cell in code] == list(range(1, len(code) + 1))
    for cell in code:
        ast.parse(cell.source)
        assert not any(out.output_type == 'error' for out in cell.outputs)
    for name, count in [('berlin_story.html', 5), ('marathon_story.html', 7)]:
        report = Report()
        report.feed((ROOT / name).read_text())
        assert set(report.anchors) <= report.ids
        assert len(report.images) == count
        for src in report.images:
            assert src.startswith('data:image/png;base64,')
            Image.open(io.BytesIO(base64.b64decode(src.split(',', 1)[1]))).verify()
    figures = list((ROOT / 'outputs').rglob('*.png'))
    assert len(figures) == 17
    for figure in figures:
        Image.open(figure).verify()
    for path in (ROOT / 'outputs').rglob('*.json'):
        json.loads(path.read_text())
    for path in (ROOT / 'outputs').rglob('*.csv'):
        with path.open(newline='') as handle:
            reader = csv.DictReader(handle)
            assert not set(reader.fieldnames or []) & {'Name', 'Zip', 'Bib', 'runner_key'}
            assert any(True for _ in reader), f'Empty table: {path}'
    manifests = list((ROOT / 'data/raw').rglob('manifest.json'))
    assert len(manifests) == 7
    for path in manifests:
        manifest = json.loads(path.read_text())
        assert manifest['version'] >= 1 and manifest['files']
        for digest in manifest['files'].values():
            assert re.fullmatch(r'[0-9a-f]{64}', digest)
    for doc in [ROOT / 'README.md', *(ROOT / 'docs').glob('*.md')]:
        for link in re.findall(r'\]\(([^)]+)\)', doc.read_text()):
            if not link.startswith(('https://', 'http://', '#')):
                assert (doc.parent / link.split('#')[0]).exists(), f'Broken link in {doc}: {link}'
    if (ROOT / '.git').exists():
        tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
        for name in filter(None, tracked):
            path = Path(name)
            assert '.venv' not in path.parts and '__pycache__' not in path.parts
            assert path.name not in {'access_token', 'kaggle.json', 'project_description.txt'}
            assert not path.name.startswith('.env')
            assert not (name.startswith('data/raw/') and path.name != 'manifest.json')
            assert (ROOT / path).stat().st_size < 10 * 1024 * 1024, f'Unexpected large tracked file: {name}'
    print(f'PASS: {len(code)} executed cells, two reports, {len(figures)} figures, '
          f'{len(manifests)} source manifests, local documentation links, and repository contents.')


if __name__ == '__main__':
    main()
