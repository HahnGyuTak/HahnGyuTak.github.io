#!/usr/bin/env python3
"""Export four anonymous, unmodified trajectories for the replay review page."""
import argparse
import json
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTIONS = [(-1, 0, 'top', 'bottom'), (1, 0, 'bottom', 'top'),
              (0, -1, 'left', 'right'), (0, 1, 'right', 'left')]
CANDIDATES = [
    (16, '한 칸 차이', '마지막 벽으로 큰 영역이 갈리고, 딱 한 칸 차이로 끝납니다.'),
    (26, '영역 쟁탈전', '후반에 여러 영역이 연달아 나뉩니다. 빈 영역도 남는 판입니다.'),
    (10, '좁은 길의 공방', '좁아지는 통로를 따라 말을 옮기며 마지막 영역을 나눕니다.'),
    (42, '긴 승부', '봇끼리 47수를 주고받은 뒤 한 칸 차이로 끝납니다.'),
]


def final_animation(state):
    owners = {(p['row'], p['col']): p['owner'] for p in state['pieces']}
    seen, territories = set(), []
    offset = 0
    counts = {'A': 0, 'B': 0}
    for row in range(7):
        for col in range(7):
            if (row, col) in seen:
                continue
            region = []

            def visit(r, c):
                seen.add((r, c))
                region.append((r, c))
                for dr, dc, side, opposite in DIRECTIONS:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < 7 and 0 <= nc < 7) or (nr, nc) in seen:
                        continue
                    if f'{r},{c}:{side}' in state['walls'] or f'{nr},{nc}:{opposite}' in state['walls']:
                        continue
                    visit(nr, nc)

            visit(row, col)
            players = {owners[cell] for cell in region if cell in owners}
            if not players:
                continue
            assert len(players) == 1, 'Terminal state has a mixed region'
            owner = players.pop()
            counts[owner] += len(region)
            membership = set(region)
            depths = {region[0]: 0}
            queue = deque([region[0]])
            while queue:
                r, c = queue.popleft()
                for dr, dc, _, _ in DIRECTIONS:
                    cell = (r + dr, c + dc)
                    if cell in membership and cell not in depths:
                        depths[cell] = depths[(r, c)] + 1
                        queue.append(cell)
            assert len(depths) == len(region)
            territories.append({'owner': owner, 'waveStartMs': offset,
                                'cells': [{'row': r, 'col': c, 'distance': depths[(r, c)]} for r, c in region]})
            offset += (max(depths.values()) + 1) * 40 + 500
    result = state['result']
    assert (counts['A'], counts['B']) == (result['aCount'], result['bCount'])
    return {'winner': result['winner'].upper(), 'aCount': counts['A'], 'bCount': counts['B'],
            'territories': territories,
            'wave': {'cellStaggerMs': 40, 'resultAtMs': offset + 200}}


def export(corpus):
    output = []
    for index, title, description in CANDIDATES:
        game = corpus['games'][index]
        frames = [{'step': 0, 'turnNumber': 0, 'actor': None, 'action': None, 'state': game['initialState']}]
        for turn in game['turns']:
            assert frames[-1]['state'] == turn['beforeState']
            step = len(frames)
            frames.append({'step': step, 'turnNumber': step, 'actor': turn['actor'],
                           'action': turn['observedAction'], 'state': turn['afterState']})
        assert frames[-1]['state'] == game['finalState']
        online = game['source']['kind'] == 'historical_recorded_online'
        data = {'schemaVersion': 1, 'boardSize': 7,
                'source': {'kind': game['source']['kind'], 'gameIndex': index,
                           'totalTurns': len(game['turns']), 'startTurnIndex': 0,
                           'participants': game['source']['participants']},
                'frames': frames, 'final': final_animation(game['finalState'])}
        output.append({'id': index, 'title': title, 'description': description,
                       'sourceLabel': '온라인 기록 · 참가자 유형 미확인' if online else '봇 벤치마크', 'data': data})
    destination = ROOT / 'assets/replays/candidates.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, separators=(',', ':')))
    print(json.dumps({'games': len(output), 'turns': [len(x['data']['frames']) - 1 for x in output],
                      'bytes': destination.stat().st_size}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('corpus', type=Path)
    export(json.loads(parser.parse_args().corpus.read_text()))
