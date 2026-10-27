#!/usr/bin/env python3
"""Package only the public Wall바둑 website for Cloudflare Pages."""

import argparse
import json
import shutil
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = [
    'index.html', 'landing.css', 'style.css',
    'replays.html', 'replays.css', 'scripts/replays.js', 'assets/replays/candidates.json',
    'support_ko.html', 'support_en.html', 'privacy.html',
    'scripts/landing.js', 'scripts/board-replay.js', 'scripts/board-replay.css',
    'assets/phone-bezel.png', 'assets/devils-plan.jpg', 'assets/replay.json',
    'assets/brand/app-store-badge.svg', 'assets/brand/google-play-badge.png',
    'assets/brand/wallbaduk-app-icon.png',
    'assets/game/ChungjuKimSaeng.otf', 'assets/game/display_bi-web.jpg',
    'assets/game/gameboard.png', 'assets/game/material-chevron-left.svg',
    'assets/game/material-schedule.svg', 'assets/game/material-icons-LICENSE.txt',
    'assets/game/playerA.png', 'assets/game/playerB.png',
    'assets/game/wallA_col.png', 'assets/game/wallA_row.png',
    'assets/game/wallB_col.png', 'assets/game/wallB_row.png',
]


def package(output: Path, base_url: str) -> None:
    parsed = urlsplit(base_url)
    if parsed.scheme != 'https' or not parsed.netloc or parsed.path not in ('', '/'):
        raise ValueError('--base-url must be an HTTPS website root')
    base_url = base_url.rstrip('/')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose an empty output directory')
    for relative in PUBLIC_FILES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(relative)
    output.mkdir(parents=True, exist_ok=True)
    for relative in PUBLIC_FILES:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    index = output / 'index.html'
    html = index.read_text()
    html = html.replace('https://hahngyutak.github.io/WallaGo', base_url)
    html = html.replace('https://wallbaduk.pages.dev', base_url)
    canonical = f'<link rel="canonical" href="{base_url}/">'
    if 'rel="canonical"' not in html:
        html = html.replace('</head>', f'    {canonical}\n  </head>')
    index.write_text(html)
    (output / '_headers').write_text(
        '/*\n  Referrer-Policy: strict-origin-when-cross-origin\n'
    )
    assert 'hahngyutak.github.io' not in html.lower()
    assert 'figmacapture' not in html
    data = json.loads((output / 'assets/replay.json').read_text())
    assert len(data['frames']) == 9
    print(json.dumps({
        'output': str(output), 'baseUrl': base_url,
        'files': len(PUBLIC_FILES) + 1,
        'bytes': sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
        'replayFrames': len(data['frames']),
    }, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-url', required=True)
    arguments = parser.parse_args()
    package(arguments.output.resolve(), arguments.base_url)
