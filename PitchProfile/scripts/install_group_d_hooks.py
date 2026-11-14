"""Connect Group D's live model view and model performance page to the app (run once, on main).

The pages are new files of Group D (football_profiler/event_live.py, football_profiler/model_performance.py,
static/event_live.js, static/model_performance.js and their .css). The app finds them through four
lines in two shared files, which this adds:
  football_profiler/app.py   import Group D's router, and include it next to the match router
  static/index.html          load the two scripts after the others
Running it again changes nothing. If a line it looks for has moved, it stops and names it; then add
the four lines by hand (README, "Connecting the pages").

Usage (from PitchProfile/): .\\.venv\\Scripts\\python.exe scripts\\install_group_d_hooks.py [--check]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# (file, line it goes after, line to add)
HOOKS = [
    ('football_profiler/app.py', 'from .match_api import router as match_router',
     'from .event_live import router as event_live_router'),
    ('football_profiler/app.py', 'app.include_router(match_router)', 'app.include_router(event_live_router)'),
    ('static/index.html', '<script src="/static/naming.js"></script>', '<script src="/static/event_live.js"></script>'),
    ('static/index.html', '<script src="/static/event_live.js"></script>',
     '<script src="/static/model_performance.js"></script>'),
]


def install(root=ROOT, check=False):
    """Add the missing hook lines, all or none; returns [(file, line added)]. With check, writes nothing.

    Files are read and written as bytes, so their line endings (LF or CRLF) stay as they are."""
    files, added = {}, []
    for file, anchor, line in HOOKS:
        if file not in files:
            files[file] = (Path(root) / file).read_bytes().decode('utf-8').splitlines(keepends=True)
        lines = files[file]
        if any(x.strip() == line for x in lines):
            continue
        at = next((i for i, x in enumerate(lines) if x.strip() == anchor), None)
        if at is None:
            raise SystemExit(f'{file}: the line "{anchor}" was not found; nothing was changed. '
                             f'Add "{line}" by hand (README, "Connecting the pages").')
        newline = '\r\n' if lines[at].endswith('\r\n') else '\n'
        if not lines[at].endswith('\n'):
            lines[at] += newline
        lines.insert(at + 1, line + newline)
        added.append((file, line))
    if not check:
        for file in dict.fromkeys(f for f, _ in added):
            (Path(root) / file).write_bytes(''.join(files[file]).encode('utf-8'))
    return added


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--check', action='store_true', help='only say what would be added')
    args = p.parse_args()
    added = install(check=args.check)
    if not added:
        print('Group D pages are already connected; nothing to do.')
    for file, line in added:
        print(f"{'would add' if args.check else 'added'} to {file}: {line}")


if __name__ == '__main__':
    main()
