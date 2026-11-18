"""The installer adds Group D's four hook lines to main's app.py and index.html once, or changes nothing."""
import shutil
from pathlib import Path

import pytest

from scripts import install_group_d_hooks as HOOKS

ROOT = Path(__file__).resolve().parents[1]


def test_installer_adds_the_four_hook_lines_once_and_keeps_line_endings(tmp_path):
    for file in ('football_profiler/app.py', 'static/index.html'):
        (tmp_path / file).parent.mkdir(parents=True, exist_ok=True)
        text = (ROOT / file).read_bytes().decode('utf-8').replace('\r\n', '\n')    # LF or CRLF checkout
        for _, _, line in HOOKS.HOOKS:                    # start from main's files, without the hooks
            text = text.replace(line + '\n', '')
        (tmp_path / file).write_bytes(text.replace('\n', '\r\n').encode('utf-8'))
    before = {f: (tmp_path / f).read_bytes() for f in ('football_profiler/app.py', 'static/index.html')}
    assert len(HOOKS.install(tmp_path, check=True)) == 4
    assert {f: (tmp_path / f).read_bytes() for f in before} == before           # --check writes nothing
    assert len(HOOKS.install(tmp_path)) == 4
    assert HOOKS.install(tmp_path) == []                                          # second run: nothing
    for file, _, line in HOOKS.HOOKS:
        text = (tmp_path / file).read_bytes().decode('utf-8')
        assert text.count(line) == 1 and '\r\n' in text and '\n' not in text.replace('\r\n', '')
    app = (tmp_path / 'football_profiler/app.py').read_text(encoding='utf-8')
    assert app.index('include_router(match_router)') < app.index('include_router(event_live_router)')
    html = (tmp_path / 'static/index.html').read_text(encoding='utf-8')
    assert html.index('naming.js') < html.index('event_live.js') < html.index('model_performance.js') < html.index('</body>')


def test_installer_changes_nothing_when_a_line_has_moved(tmp_path):
    for file in ('football_profiler/app.py', 'static/index.html'):
        (tmp_path / file).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / file, tmp_path / file)
    app = tmp_path / 'football_profiler/app.py'
    app.write_text(app.read_text(encoding='utf-8').replace('from .event_live import router as event_live_router\n', '')
                   .replace('app.include_router(event_live_router)\n', '')
                   .replace('app.include_router(match_router)', 'app.include_router(match_router, prefix="")'), encoding='utf-8')
    html = (tmp_path / 'static/index.html').read_bytes()
    with pytest.raises(SystemExit):
        HOOKS.install(tmp_path)
    assert (tmp_path / 'static/index.html').read_bytes() == html
