/**
 * A recorded Wall Baduk endgame, rendered with the Flutter app's own assets.
 * The host owns playback: call setStep(0..state.totalSteps).
 * Include board-replay.css alongside this module.
 */
export async function mountBoardReplay(root, options = {}) {
  if (!root || typeof root.replaceChildren !== 'function') {
    throw new TypeError('mountBoardReplay needs a DOM element.');
  }

  let data = options.data;
  if (!data) {
    const dataUrl = options.dataUrl || new URL('../assets/replay.json', import.meta.url);
    const response = await fetch(dataUrl);
    if (!response.ok) throw new Error(`Replay could not load (${response.status}).`);
    data = await response.json();
  }
  if (data.schemaVersion !== 1 || data.boardSize !== 7 || !Array.isArray(data.frames) || data.frames.length < 2) {
    throw new Error('Unsupported replay data.');
  }
  const totalSteps = data.frames.length - 1;
  const winner = data.final.winner.toUpperCase();
  const winnerName = winner === 'A' ? '빨강' : winner === 'B' ? '파랑' : '무승부';
  const finalDescription = winner === 'A' || winner === 'B'
    ? `${winnerName} 승리. 빨강 ${data.final.aCount}칸, 파랑 ${data.final.bCount}칸.`
    : `무승부. 빨강 ${data.final.aCount}칸, 파랑 ${data.final.bCount}칸.`;

  const assetDirectory = String(options.assetBase || new URL('../assets/game/', import.meta.url));
  const assetBase = new URL(assetDirectory.endsWith('/') ? assetDirectory : `${assetDirectory}/`, document.baseURI);
  const asset = (name) => new URL(name, assetBase).href;
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const timers = new Set();
  const stoneElements = [];
  const wallElements = new Map();
  const cellElements = new Map();
  const state = {
    step: 0, frame: data.frames[0], ended: false, animating: false, totalSteps,
    startTurn: data.frames[0].turnNumber, endTurn: data.frames[totalSteps].turnNumber,
  };
  let destroyed = false;

  const element = (tag, className, text) => {
    const node = document.createElement(tag);
    node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const picture = (name, className) => {
    const node = element('img', className);
    node.src = asset(name);
    node.alt = '';
    node.draggable = false;
    node.decoding = 'async';
    return node;
  };
  const later = (callback, delay) => {
    const timer = window.setTimeout(() => {
      timers.delete(timer);
      if (!destroyed) callback();
    }, delay);
    timers.add(timer);
  };
  const clearTimers = () => {
    for (const timer of timers) window.clearTimeout(timer);
    timers.clear();
  };

  const app = element('div', 'replay-app');
  app.dataset.reducedMotion = String(reducedMotion);
  const header = element('div', 'replay-header');
  // Official Material icons, matching Flutter's Material icon family.
  const back = picture('material-chevron-left.svg', 'replay-back');
  back.setAttribute('aria-hidden', 'true');
  header.append(back, element('p', 'replay-title', '온라인 대결'));

  const match = element('div', 'replay-match');
  const playerA = element('div', 'replay-player replay-player-a');
  playerA.append(picture('playerA.png', 'replay-avatar'), element('span', 'replay-player-name', '빨강'));
  const playerB = element('div', 'replay-player replay-player-b');
  playerB.append(element('span', 'replay-player-name', '파랑'), picture('playerB.png', 'replay-avatar'));
  match.append(playerA, element('span', 'replay-versus', 'vs'), playerB);

  const stage = element('div', 'replay-stage');
  const scoreboard = element('div', 'replay-scoreboard');
  scoreboard.style.backgroundImage = `url("${asset('display_bi-web.jpg')}")`;
  const clock = element('div', 'replay-clock');
  const clockIcon = picture('material-schedule.svg', 'replay-clock-icon');
  clockIcon.setAttribute('aria-hidden', 'true');
  clock.append(clockIcon, element('span', 'replay-seconds', '60'), element('span', 'replay-seconds-unit', '초'));
  const instruction = element('div', 'replay-instruction');
  const turnStone = picture('playerA.png', 'replay-turn-stone');
  const instructionText = element('span', 'replay-instruction-text', '말을 이동하세요');
  instruction.append(turnStone, instructionText);
  scoreboard.append(clock, instruction);

  const board = element('div', 'replay-board');
  board.setAttribute('role', 'img');
  board.append(picture('gameboard.png', 'replay-board-image'));
  const grid = element('div', 'replay-grid');
  grid.setAttribute('aria-hidden', 'true');
  for (let row = 0; row < 7; row += 1) {
    for (let col = 0; col < 7; col += 1) {
      const cell = element('span', 'replay-cell');
      cell.style.left = `${col / 7 * 100}%`;
      cell.style.top = `${row / 7 * 100}%`;
      cellElements.set(`${row},${col}`, cell);
      grid.append(cell);
    }
  }
  const wallsLayer = element('div', 'replay-walls');
  const stonesLayer = element('div', 'replay-stones');
  grid.append(wallsLayer, stonesLayer);
  board.append(grid);

  const result = element('div', 'replay-result');
  result.hidden = true;
  const winnerLine = element('div', 'replay-winner');
  if (winner === 'A' || winner === 'B') winnerLine.append(picture(`player${winner}.png`, 'replay-winner-stone'));
  winnerLine.append(element('span', 'replay-victory', winner === 'A' || winner === 'B' ? '승리!' : '무승부'));
  const resultScore = element('div', 'replay-final-score');
  resultScore.append(
    picture('playerA.png', 'replay-score-stone'),
    element('span', 'replay-final-a', String(data.final.aCount)),
    element('span', 'replay-score-vs', 'vs'),
    element('span', 'replay-final-b', String(data.final.bCount)),
    picture('playerB.png', 'replay-score-stone'),
  );
  result.append(winnerLine, resultScore);
  board.append(result);
  stage.append(scoreboard, board);
  const live = element('span', 'replay-sr-only');
  live.setAttribute('role', 'status');
  live.setAttribute('aria-live', 'polite');
  app.append(header, match, stage, live);
  root.classList.add('replay-root');
  root.replaceChildren(app);

  function resetVictory() {
    result.hidden = true;
    for (const cell of cellElements.values()) {
      cell.className = 'replay-cell';
    }
    state.animating = false;
    app.classList.remove('replay-ended');
  }

  function showVictory() {
    if (destroyed || state.step !== totalSteps) return;
    result.hidden = false;
    state.animating = false;
    app.classList.add('replay-ended');
    live.textContent = finalDescription;
  }

  function revealTerritories() {
    state.animating = true;
    for (const territory of data.final.territories) {
      for (const cellData of territory.cells) {
        const cell = cellElements.get(`${cellData.row},${cellData.col}`);
        const reveal = () => cell.classList.add(`replay-cell-${territory.owner.toLowerCase()}`, 'replay-cell-visible');
        if (reducedMotion) reveal();
        else later(reveal, territory.waveStartMs + cellData.distance * data.final.wave.cellStaggerMs);
      }
    }
    if (reducedMotion) showVictory();
    else later(showVictory, data.final.wave.resultAtMs);
  }

  function drawStones(frame, animate) {
    frame.state.pieces.forEach((piece, index) => {
      let stone = stoneElements[index];
      if (!stone) {
        stone = picture(`player${piece.owner}.png`, 'replay-piece');
        stonesLayer.append(stone);
        stoneElements[index] = stone;
      }
      stone.style.transitionDuration = animate && !reducedMotion ? '300ms' : '0ms';
      stone.style.left = `${(piece.col + 0.5) / 7 * 100}%`;
      stone.style.top = `${(piece.row + 0.5) / 7 * 100}%`;
      stone.dataset.row = String(piece.row);
      stone.dataset.col = String(piece.col);
      stone.dataset.owner = piece.owner;
    });
  }

  function drawWalls(frame, animate) {
    const present = new Set();
    for (const [key, owner] of Object.entries(frame.state.walls)) {
      const [position, direction] = key.split(':');
      const [row, col] = position.split(',').map(Number);
      const horizontal = direction === 'top' || direction === 'bottom';
      const boundary = horizontal ? row + (direction === 'bottom' ? 1 : 0) : col + (direction === 'right' ? 1 : 0);
      const wallId = horizontal ? `h:${boundary}:${col}` : `v:${row}:${boundary}`;
      if (present.has(wallId)) continue;
      present.add(wallId);
      let wall = wallElements.get(wallId);
      if (!wall) {
        wall = picture(`wall${owner}_${horizontal ? 'col' : 'row'}.png`, `replay-wall replay-wall-${horizontal ? 'horizontal' : 'vertical'}`);
        wall.style.left = `${(horizontal ? col + 0.5 : boundary) / 7 * 100}%`;
        wall.style.top = `${(horizontal ? boundary : row + 0.5) / 7 * 100}%`;
        wall.dataset.wall = wallId;
        wall.dataset.owner = owner;
        if (animate && !reducedMotion) wall.classList.add('replay-wall-new');
        wallsLayer.append(wall);
        wallElements.set(wallId, wall);
      } else {
        // Finish an interrupted entrance before starting the next recorded turn.
        wall.classList.remove('replay-wall-new');
      }
    }
    for (const [key, wall] of wallElements) {
      if (!present.has(key)) {
        wall.remove();
        wallElements.delete(key);
      }
    }
  }

  function render(frame, animate) {
    drawStones(frame, animate);
    drawWalls(frame, animate);
    const ended = frame.state.phase === 'ended';
    turnStone.src = asset(`player${frame.state.turn}.png`);
    instructionText.textContent = ended ? '영역을 확인하세요' : '말을 이동하세요';
    board.setAttribute('aria-label', `대국 ${frame.turnNumber}수. ${ended ? finalDescription : `${frame.state.turn === 'A' ? '빨강' : '파랑'} 차례.`}`);
    board.dataset.step = String(frame.step);
    board.dataset.turnNumber = String(frame.turnNumber);
    root.dataset.step = String(frame.step);
    root.dataset.turnNumber = String(frame.turnNumber);
    live.textContent = `${frame.turnNumber}수. ${ended ? '대국 종료. 영역 계산 중.' : `${frame.state.turn === 'A' ? '빨강' : '파랑'} 차례.`}`;
  }

  function setStep(requestedStep) {
    if (destroyed) return state;
    const parsed = Number(requestedStep);
    const step = Number.isFinite(parsed) ? Math.min(totalSteps, Math.max(0, Math.round(parsed))) : 0;
    if (step === state.step) return state;
    clearTimers();
    resetVictory();
    const animate = step > state.step;
    state.step = step;
    state.frame = data.frames[step];
    state.ended = state.frame.state.phase === 'ended';
    render(state.frame, animate);
    if (state.ended) {
      state.animating = true;
      if (reducedMotion) revealTerritories();
      else later(revealTerritories, 360);
    }
    return state;
  }

  function destroy() {
    if (destroyed) return;
    destroyed = true;
    clearTimers();
    state.animating = false;
    app.remove();
    root.classList.remove('replay-root');
    delete root.dataset.step;
    delete root.dataset.turnNumber;
  }

  render(state.frame, false);
  return { setStep, destroy, state };
}
