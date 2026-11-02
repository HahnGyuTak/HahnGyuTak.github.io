// Presentation ports of MoveStepPath, PieceMoveSequence and TerritoryRevealPlan.
// The recorded game states and scores remain authoritative.
export const MOTION = Object.freeze({ stepMs: 150, depthMs: 200, fadeMs: 400, gapMs: 250 });
const directions = [[-1, 0, 'top', 'bottom'], [1, 0, 'bottom', 'top'], [0, -1, 'left', 'right'], [0, 1, 'right', 'left']];
const key = (row, col) => `${row},${col}`;
const inside = (row, col) => row >= 0 && row < 7 && col >= 0 && col < 7;

function neighbors(row, col, walls) {
  return directions.flatMap(([dr, dc, side, opposite]) => {
    const nr = row + dr, nc = col + dc;
    return inside(nr, nc) && !Object.hasOwn(walls, `${row},${col}:${side}`) && !Object.hasOwn(walls, `${nr},${nc}:${opposite}`) ? [[nr, nc]] : [];
  });
}

export function planMove(before, action, actor) {
  if (!action) return [];
  const [fr, fc] = action.from, [tr, tc] = action.to;
  if (!before.pieces.some((p) => p.owner === actor && p.row === fr && p.col === fc)) return null;
  if (fr === tr && fc === tc) return [];
  const occupied = new Set(before.pieces.map((p) => key(p.row, p.col)));
  const canStep = (r, c, nr, nc) => !occupied.has(key(nr, nc)) && neighbors(r, c, before.walls).some(([a, b]) => a === nr && b === nc);
  if (canStep(fr, fc, tr, tc)) return [[tr, tc]];
  for (const [dr, dc] of directions) {
    const mr = fr + dr, mc = fc + dc;
    if (canStep(fr, fc, mr, mc) && canStep(mr, mc, tr, tc)) return [[mr, mc], [tr, tc]];
  }
  return null;
}

export function movePosition(from, steps, progress) {
  if (!steps.length || !(progress > 0)) return from;
  if (progress >= 1) return steps.at(-1);
  const elapsed = progress * steps.length, index = Math.floor(elapsed);
  const start = index === 0 ? from : steps[index - 1], end = steps[index];
  const eased = 1 - (1 - (elapsed - index)) ** 3;
  return [start[0] + (end[0] - start[0]) * eased, start[1] + (end[1] - start[1]) * eased];
}

export function territoryStages(board, first = 'A') {
  const occupied = new Map(board.pieces.map((p) => [key(p.row, p.col), p.owner]));
  const seen = new Set(), cells = { A: [], B: [] };
  for (let row = 0; row < 7; row += 1) for (let col = 0; col < 7; col += 1) {
    if (seen.has(key(row, col))) continue;
    const queue = [[row, col]], region = [];
    seen.add(key(row, col));
    for (let i = 0; i < queue.length; i += 1) {
      const [r, c] = queue[i]; region.push([r, c]);
      for (const [nr, nc] of neighbors(r, c, board.walls)) if (!seen.has(key(nr, nc))) {
        seen.add(key(nr, nc)); queue.push([nr, nc]);
      }
    }
    const owners = new Set(region.map(([r, c]) => occupied.get(key(r, c))).filter(Boolean));
    if (owners.size !== 1) continue;
    const owner = [...owners][0];
    const membership = new Set(region.map(([r, c]) => key(r, c)));
    const seeds = board.pieces.filter((p) => membership.has(key(p.row, p.col)));
    const depths = new Map(seeds.map((p) => [key(p.row, p.col), 0]));
    const frontier = seeds.map((p) => [p.row, p.col]);
    for (let i = 0; i < frontier.length; i += 1) {
      const [r, c] = frontier[i];
      for (const [nr, nc] of neighbors(r, c, board.walls)) {
        const next = key(nr, nc);
        if (membership.has(next) && !depths.has(next)) {
          depths.set(next, depths.get(key(r, c)) + 1); frontier.push([nr, nc]);
        }
      }
    }
    for (const [r, c] of region) cells[owner].push({ row: r, col: c, depth: depths.get(key(r, c)) });
  }
  const order = first === 'B' ? ['B', 'A'] : ['A', 'B'];
  let elapsed = 0;
  return order.filter((owner) => cells[owner].length).map((owner, index) => {
    if (index) elapsed += MOTION.gapMs;
    const startMs = elapsed;
    const maxDepth = Math.max(...cells[owner].map((cell) => cell.depth));
    elapsed += maxDepth * MOTION.depthMs + MOTION.fadeMs;
    return { owner, cells: cells[owner], startMs, endMs: elapsed };
  });
}
