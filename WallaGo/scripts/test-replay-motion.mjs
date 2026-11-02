import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { MOTION, planMove, movePosition, territoryStages } from './replay-motion.js';

assert.deepEqual(MOTION, { stepMs: 150, depthMs: 200, fadeMs: 400, gapMs: 250 });
assert.deepEqual(movePosition([0, 0], [[0, 1], [1, 1]], 0), [0, 0]);
assert.deepEqual(movePosition([0, 0], [[0, 1], [1, 1]], .25), [0, .875]);
assert.deepEqual(movePosition([0, 0], [[0, 1], [1, 1]], .5), [0, 1]);
assert.deepEqual(movePosition([0, 0], [[0, 1], [1, 1]], 1), [1, 1]);
const blocked = { pieces: [{ owner: 'A', row: 1, col: 1 }], walls: { '1,1:right': 'A' } };
assert.deepEqual(planMove(blocked, { from: [1, 1], to: [2, 2] }, 'A'), [[2, 1], [2, 2]]);
assert.equal(planMove(blocked, { from: [1, 1], to: [1, 3] }, 'A'), null);

const byCell = (cells) => [...cells].sort((a, b) => a.row - b.row || a.col - b.col);
let moves = 0;
if (process.argv[2]) {
  const references = JSON.parse(readFileSync(process.argv[2], 'utf8'));
  for (const { data, paths, stages } of references) {
    for (let i = 1; i < data.frames.length; i += 1) {
      const frame = data.frames[i];
      assert.deepEqual(planMove(data.frames[i - 1].state, frame.action, frame.actor), paths[i - 1], `path: turn ${frame.turnNumber}`);
      moves += 1;
    }
    for (const owner of ['A', 'B']) {
      const actual = territoryStages(data.frames.at(-1).state, owner);
      assert.equal(actual.length, stages[owner].length);
      actual.forEach((stage, index) => assert.deepEqual({ ...stage, cells: byCell(stage.cells) }, { ...stages[owner][index], cells: byCell(stages[owner][index].cells) }));
      for (const color of ['A', 'B']) assert.equal(actual.find((stage) => stage.owner === color)?.cells.length || 0, data.final[color === 'A' ? 'aCount' : 'bCount']);
    }
  }
  console.log(`PASS: ${moves} recorded move paths and both reveal orders for ${references.length} replays match the actual Dart app.`);
} else {
  const data = JSON.parse(readFileSync(new URL('../assets/replay.json', import.meta.url), 'utf8'));
  const stages = territoryStages(data.frames.at(-1).state);
  assert.equal(stages.find((s) => s.owner === 'A').cells.length, 25);
  assert.equal(stages.find((s) => s.owner === 'B').cells.length, 24);
  assert.equal(stages[1].startMs - stages[0].endMs, 250);
  console.log('PASS: path, easing, selected replay ownership and reveal timing.');
}
