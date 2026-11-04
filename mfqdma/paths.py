"""Repository paths and data helpers (no hard-coded absolute paths)."""
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / 'results'
FIGURES = REPO / 'figures_out'          # figures produced by the notebooks
CACHE = REPO / 'cache'                  # cached intermediate computations
DATA = REPO / 'data'
QRM_REPO = DATA / 'qrmdata'             # git clone of github.com/cran/qrmdata
QRM_DATA = QRM_REPO / 'data'
BONN = DATA / 'bonn'                    # Bonn EEG: folders Z, O, N, F, S with .txt files

for _d in (FIGURES, CACHE, DATA, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)


def ensure_qrmdata():
    """Clone the CRAN mirror of qrmdata (Hofert, Hornik & McNeil) on first use.
    The .rda files are read with mfqdma.rdata_reader, a pure-data parser."""
    if not (QRM_DATA / 'SP500.rda').exists():
        print('cloning https://github.com/cran/qrmdata ...', flush=True)
        subprocess.run(['git', 'clone', '--depth', '1',
                        'https://github.com/cran/qrmdata', str(QRM_REPO)], check=True)
    return QRM_DATA


def ensure_bonn():
    if not (BONN / 'Z').is_dir():
        raise FileNotFoundError(
            'Bonn EEG data not found. Download the five sets (Z, O, N, F, S) from\n'
            'https://www.ukbonn.de/epileptologie/arbeitsgruppen/ag-lehnertz-neurophysik/downloads/\n'
            f'and extract them so that {BONN}/Z/*.txt ... {BONN}/S/*.txt exist.')
    return BONN


def prepare_bonn(*sources):
    """Put the Bonn EEG sets into data/bonn/{Z,O,N,F,S}.

    Each source may be a folder (searched recursively), or an archive (.zip,
    .tar, .tar.gz, or .rar if the `unrar` program is installed). Both layouts
    in circulation are recognized: folders named Z, O, N, F, S (e.g. the
    archive Dataset.rar with Dataset/Z/Z001.txt ...), and the five official
    zip files z.zip ... s.zip whose text files are named Z001.txt, O001.txt,
    N001.TXT, F001.txt, S001.txt.
    """
    import shutil
    import tempfile
    sets = ['Z', 'O', 'N', 'F', 'S']
    tmp = Path(tempfile.mkdtemp(dir=DATA))
    roots = []
    for k, src in enumerate(sources):
        src = Path(src).expanduser()
        if src.is_dir():
            roots.append(src)
            continue
        dest = tmp / f'src{k}'
        dest.mkdir()
        if src.suffix.lower() == '.rar':
            if shutil.which('unrar') is None:
                raise RuntimeError('unrar not installed: in Colab run !apt-get -qq install unrar, '
                                   'or extract the archive yourself and pass the folder.')
            subprocess.run(['unrar', 'x', '-o+', '-inul', str(src), str(dest) + '/'], check=True)
        else:
            shutil.unpack_archive(str(src), str(dest))
        roots.append(dest)
    found = {}
    for root in roots:
        for f in list(root.rglob('*.txt')) + list(root.rglob('*.TXT')):
            s = f.name[0].upper()
            if s in sets and f.stem[1:].isdigit():
                found.setdefault(s, set()).add(f)
    for s in sets:
        files = sorted(found.get(s, ()))
        if len(files) != 100:
            raise FileNotFoundError(f'set {s}: found {len(files)} files, expected 100')
        (BONN / s).mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy(f, BONN / s / f.name)
    shutil.rmtree(tmp, ignore_errors=True)
    print('Bonn EEG ready in', BONN, {s: len(found[s]) for s in sets})
    return BONN
