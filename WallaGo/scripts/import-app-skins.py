#!/usr/bin/env python3
"""Copy the app's actual theme assets and visibility policy, without image edits."""
import argparse
import json
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def import_skins(app, lossless_webp=False):
    source = (app / 'lib/services/game_themes.dart').read_text()
    flags = (app / 'lib/services/app_feature_flags.dart').read_text()
    preview = re.search(r'previewThemeIds\s*=\s*<String>\{([^}]+)\}', source).group(1)
    visible = set(re.findall(r"'([^']+)'", preview))
    full = bool(re.search(r'fullThemeCatalogEnabled\s*=\s*true', flags))
    themes = []
    copies = []
    extension = 'webp' if lossless_webp else 'png'
    mapping = {'playerA': 'playerAImagePath', 'playerB': 'playerBImagePath',
               'wallA_row': 'wallARowImagePath', 'wallA_col': 'wallAColImagePath',
               'wallB_row': 'wallBRowImagePath', 'wallB_col': 'wallBColImagePath'}
    for block in re.findall(r'GameTheme\(\s*\n(.*?)\n  \)', source, re.S):
        def field(name):
            return re.search(rf"{name}:\s*'([^']+)'", block).group(1)
        theme_id = field('id')
        assets = {}
        for target, prop in mapping.items():
            relative = f'assets/skins/{theme_id}/{target}.{extension}'
            destination = ROOT / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            copies.append((app / field(prop), destination))
            assets[target] = relative
        for owner in ['A', 'B']:
            relative = f'assets/skins/{theme_id}/territory{owner}.{extension}'
            copies.append((app / f'lib/img/territory/{theme_id}_{owner.lower()}.png', ROOT / relative))
            assets[f'territory{owner}'] = relative
        themes.append({'id': theme_id, 'name': field('nameKo'),
                       'available': full or theme_id in visible, 'assets': assets})
    assert len(themes) == 6
    def convert(pair):
        source, target = pair
        if lossless_webp:
            subprocess.run(['cwebp', '-quiet', '-lossless', '-exact', '-m', '4', str(source), '-o', str(target)], check=True)
        else:
            shutil.copy2(source, target)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(convert, copies))
    (ROOT / 'assets/themes.json').write_text(json.dumps(themes, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'themes': [t['name'] for t in themes], 'visible': [t['id'] for t in themes if t['available']], 'files': len(themes) * 8}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('app', type=Path)
    parser.add_argument('--lossless-webp', action='store_true', help='Use cwebp lossless/exact, preserving every RGBA pixel')
    args = parser.parse_args()
    import_skins(args.app, args.lossless_webp)
