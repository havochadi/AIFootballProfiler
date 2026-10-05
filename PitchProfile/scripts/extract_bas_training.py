"""Unpack the password-protected SoccerNet training videos on the data drive.

Covers SoccerNet Ball Action Spotting (sn-bas-2025: train, valid, test) and FOOTPASS
player-centric action spotting (sn-pcbas-2026: 640x352 videos). The zips are AES-protected
with the password SoccerNet sends after its NDA is signed. Run this in a terminal: it asks for
the password without echoing it and never stores or prints it. The password is checked on one
file before anything is extracted. Zips not downloaded yet are listed and skipped; already
extracted zips are skipped, so the command can be rerun after more downloads finish.

Usage (from PitchProfile/, in your own terminal):
  .\\.venv\\Scripts\\python.exe scripts\\extract_bas_training.py
"""
from __future__ import annotations

import getpass
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from football_profiler import soccernet as SN  # noqa: E402


def archives():
    """(zip path, output folder) for every password-protected video archive we use."""
    bas, fp = SN.root() / 'sn-bas-2025', SN.root() / 'sn-pcbas-2026'
    out = [(bas / f'{s}.zip', bas / s) for s in ('train', 'valid', 'test')]
    out += [(fp / f'videos_352x640_{s}.zip', fp / 'videos_352x640' / s) for s in ('TRAIN', 'VAL')]
    # Full-HD FOOTPASS: identity ground truth (player boxes with shirt numbers) at broadcast resolution.
    out += [(fp / 'videos_fullHD_VAL.zip', fp / 'videos_fullHD' / 'VAL')]
    out += [(fp / f'videos_fullHD_TRAIN_{k:02d}.zip', fp / 'videos_fullHD' / 'TRAIN') for k in range(1, 6)]
    return out


def done(path, out):
    """Several archives can unpack into one folder, so each has its own marker."""
    return (out / f'.extracted_{path.stem}').is_file() or \
        (out / '.extracted').is_file() and json.loads((out / '.extracted').read_text()).get('zip') == path.name


def opener(path):
    """zipfile for classic zip encryption, pyzipper for AES."""
    archive = zipfile.ZipFile(path)
    if any(i.compress_type == 99 for i in archive.infolist()):     # 99 = AES (WinZip)
        archive.close()
        try:
            import pyzipper
        except ImportError:
            raise SystemExit('These zips use AES encryption. Install the reader first:\n'
                             '  .\\.venv\\Scripts\\python.exe -m pip install pyzipper')
        return pyzipper.AESZipFile(path)
    return archive


def main():
    todo, missing = [], []
    for path, out in archives():
        if not path.is_file():
            missing.append(path.name)
        elif not done(path, out):
            todo.append((path, out))
    if missing:
        print('Not downloaded yet (skipped):', ', '.join(missing))
    if not todo:
        raise SystemExit('Nothing to extract.')
    password = getpass.getpass('SoccerNet NDA password (not shown): ').encode()
    with opener(todo[0][0]) as archive:
        small = min((i for i in archive.infolist() if not i.is_dir()), key=lambda i: i.file_size)
        try:
            archive.read(small.filename, pwd=password)
        except RuntimeError:
            raise SystemExit('That password did not open the archive. Nothing was extracted.')
    for path, out in todo:
        with opener(path) as archive:
            members = [i for i in archive.infolist() if not i.is_dir()]
            print(f'{path.name}: extracting {len(members)} files ...', flush=True)
            try:
                archive.extractall(out, pwd=password)
            except RuntimeError:
                print(f'  {path.name}: the password did not open this archive; skipped.', flush=True)
                continue
        (out / f'.extracted_{path.stem}').write_text(json.dumps({'zip': path.name, 'files': len(members)}),
                                                     encoding='utf-8')
    print('Done. You can close this terminal and tell Claude it is unpacked.')


if __name__ == '__main__':
    main()
