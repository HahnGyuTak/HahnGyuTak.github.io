#!/usr/bin/env python3
"""Package only the public Wall바둑 website for Cloudflare Pages."""

import argparse
import json
import shutil
from pathlib import Path
from urllib.parse import urlsplit
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
INDEXED_PAGES = {
    'index.html': '/',
    'devils-plan-wall-baduk/index.html': '/devils-plan-wall-baduk/',
}
PUBLIC_FILES = [
    *INDEXED_PAGES, 'devils-plan-wall-baduk/guide.css', 'landing.css', 'style.css',
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
    'skins.css', 'scripts/replay-motion.js', 'scripts/skin-showcase.js', 'assets/themes.json',
]
for theme in json.loads((ROOT / 'assets/themes.json').read_text()):
    PUBLIC_FILES.extend(theme['assets'].values())


def package(output: Path, base_url: str) -> None:
    parsed = urlsplit(base_url)
    if parsed.scheme != 'https' or not parsed.netloc or parsed.path not in ('', '/'):
        raise ValueError('--base-url must be an HTTPS website root')
    base_url = base_url.rstrip('/')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose an empty output directory')
    sources = {relative: ROOT / relative for relative in PUBLIC_FILES}
    sources['index.html'] = ROOT / '_cloudflare/index.html'
    for source in sources.values():
        if not source.is_file():
            raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=True)
    for relative, source in sources.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for relative, route in INDEXED_PAGES.items():
        page = output / relative
        html = page.read_text(encoding='utf-8')
        html = html.replace('https://hahngyutak.github.io/WallaGo', base_url)
        html = html.replace('https://wallbaduk.pages.dev', base_url)
        canonical = f'<link rel="canonical" href="{base_url}{route}">'
        assert canonical in html, f'Canonical does not match {relative}'
        assert 'hahngyutak.github.io' not in html.lower()
        assert 'figmacapture' not in html
        page.write_text(html, encoding='utf-8')
    (output / '_headers').write_text(
        '/*\n  Referrer-Policy: strict-origin-when-cross-origin\n'
    )
    # These files belong at the Pages domain root, not the GitHub blog root.
    (output / 'robots.txt').write_text(
        f'User-agent: *\nAllow: /\n\nSitemap: {base_url}/sitemap.xml\n'
    )
    urls = ''.join(
        f'  <url><loc>{escape(base_url + route)}</loc></url>\n'
        for route in INDEXED_PAGES.values()
    )
    (output / 'sitemap.xml').write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f'{urls}'
        '</urlset>\n'
    )
    data = json.loads((output / 'assets/replay.json').read_text())
    assert len(data['frames']) == 9
    print(json.dumps({
        'output': str(output), 'baseUrl': base_url,
        'files': len(PUBLIC_FILES) + 3,
        'bytes': sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
        'replayFrames': len(data['frames']),
    }, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-url', required=True)
    arguments = parser.parse_args()
    package(arguments.output.resolve(), arguments.base_url)
