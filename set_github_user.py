"""Fill in your GitHub location in the README and in the notebook.

    python set_github_user.py <github-user> [--repo <repository>] [--subdir <path>]

<repository> defaults to the name of this folder. Give --subdir only if this
folder is not the root of the repository, for example when both companion
folders sit in one repository:

    python set_github_user.py alice --repo MFQDMA --subdir MFQDMA_main
"""
import argparse
import json
from pathlib import Path

root = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('user')
ap.add_argument('--repo', default=root.name)
ap.add_argument('--subdir', default='')
a = ap.parse_args()

PH = '<YOUR-GITHUB-USERNAME>'
default_repo = root.name
sub = a.subdir.strip('/')


def fix(text):
    # Colab badge links: .../github/<user>/<repo>/blob/main/[<subdir>/]<notebook>
    text = text.replace(f'github/{PH}/{default_repo}/blob/main/',
                        f'github/{a.user}/{a.repo}/blob/main/' + (sub + '/' if sub else ''))
    # clone URL and plain links
    text = text.replace(f'github.com/{PH}/{default_repo}', f'github.com/{a.user}/{a.repo}')
    if sub:
        text = text.replace("SUBDIR = ''", f"SUBDIR = '{sub}'")
    return text.replace(PH, a.user)


readme = root / 'README.md'
readme.write_text(fix(readme.read_text()))
print('updated README.md')
for nb in sorted(root.glob('*.ipynb')):
    d = json.loads(nb.read_text())
    for c in d['cells']:
        c['source'] = fix(''.join(c['source'])).splitlines(keepends=True)
    nb.write_text(json.dumps(d, indent=1, ensure_ascii=False) + '\n')
    print('updated', nb.name)
