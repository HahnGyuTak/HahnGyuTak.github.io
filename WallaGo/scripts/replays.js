import { mountBoardReplay } from './board-replay.js';

const $ = (selector) => document.querySelector(selector);
const root = $('[data-review-board]');
const range = $('#review-progress');
const play = $('[data-play]');
const previous = $('[data-prev]');
const next = $('[data-next]');
const final = $('[data-final]');
const speed = $('[data-speed]');
const hint = $('[data-playback-hint]');
let games = [];
let selected;
let replay;
let mode = 'ending';
let step = 0;
let start = 0;
let timer;
let generation = 0;

function pause() {
  clearTimeout(timer);
  timer = undefined;
  play.textContent = '재생';
  play.setAttribute('aria-pressed', 'false');
}

function render() {
  const total = selected.data.frames.length - 1;
  replay.setStep(step);
  range.min = String(start);
  range.max = String(total);
  range.value = String(step);
  range.setAttribute('aria-valuetext', `${total}수 중 ${step}수`);
  $('[data-turn-count]').textContent = `${step} / ${total}수`;
  $('[data-range-label]').textContent = mode === 'ending' ? `${start + 1}–${total}수 · 마지막 8수` : '처음부터 전체 기보';
  previous.disabled = step <= start;
  next.disabled = step >= total;
  hint.textContent = step === total ? '영역 계산 후 승리 애니메이션이 나옵니다.' : '재생을 누르거나 슬라이더를 움직여 보세요.';
}

function go(requested) {
  pause();
  step = Math.max(start, Math.min(selected.data.frames.length - 1, requested));
  render();
}

function tick() {
  if (document.hidden) { pause(); return; }
  const total = selected.data.frames.length - 1;
  if (step >= total) { pause(); return; }
  step += 1;
  render();
  if (step >= total) pause();
  else timer = setTimeout(tick, Number(speed.value));
}

async function selectGame(id, updateUrl = true) {
  pause();
  const token = ++generation;
  selected = games.find((game) => game.id === Number(id)) || games[0];
  replay?.destroy();
  replay = undefined;
  range.disabled = true;
  for (const button of [play, previous, next, final]) button.disabled = true;
  document.querySelectorAll('[data-game]').forEach((button) => button.setAttribute('aria-pressed', String(Number(button.dataset.game) === selected.id)));
  const candidateNumber = games.indexOf(selected) + 1;
  $('[data-game-title]').textContent = `후보 ${candidateNumber} · ${selected.title}`;
  $('[data-game-description]').textContent = selected.description;
  $('[data-source]').textContent = `${selected.sourceLabel} · games[${selected.id}] · 점수는 빨강 : 파랑`;
  const instance = await mountBoardReplay(root, { data: selected.data, assetBase: './assets/game/' });
  if (token !== generation) { instance.destroy(); return; }
  replay = instance;
  start = mode === 'ending' ? Math.max(0, replay.state.totalSteps - 8) : 0;
  step = start;
  range.disabled = false;
  play.disabled = false;
  final.disabled = false;
  render();
  if (updateUrl) {
    const url = new URL(location.href);
    url.searchParams.set('game', String(selected.id));
    history.replaceState(null, '', url);
  }
}

play.addEventListener('click', () => {
  if (!replay) return;
  if (timer !== undefined) { pause(); return; }
  if (step >= replay.state.totalSteps) { step = start; render(); }
  play.textContent = '일시정지';
  play.setAttribute('aria-pressed', 'true');
  timer = setTimeout(tick, Number(speed.value));
});
previous.addEventListener('click', () => go(step - 1));
next.addEventListener('click', () => go(step + 1));
final.addEventListener('click', () => go(replay.state.totalSteps));
range.addEventListener('input', () => go(Number(range.value)));
speed.addEventListener('change', () => {
  if (timer === undefined) return;
  clearTimeout(timer);
  timer = setTimeout(tick, Number(speed.value));
});
document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });
document.querySelectorAll('[data-mode]').forEach((button) => button.addEventListener('click', () => {
  if (!replay) return;
  mode = button.dataset.mode;
  document.querySelectorAll('[data-mode]').forEach((item) => item.setAttribute('aria-pressed', String(item === button)));
  start = mode === 'ending' ? Math.max(0, replay.state.totalSteps - 8) : 0;
  go(start);
}));

try {
  const response = await fetch('./assets/replays/candidates.json');
  if (!response.ok) throw new Error(`기보 응답 ${response.status}`);
  games = await response.json();
  const list = $('[data-candidates]');
  games.forEach((game, index) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'candidate-card';
    button.dataset.game = String(game.id);
    button.setAttribute('aria-pressed', 'false');
    const number = document.createElement('span');
    number.className = 'candidate-number';
    number.textContent = `후보 ${index + 1}`;
    const title = document.createElement('strong');
    title.textContent = game.title;
    const score = document.createElement('span');
    score.className = 'candidate-score';
    score.textContent = `${game.data.final.aCount} : ${game.data.final.bCount}`;
    const length = document.createElement('span');
    length.className = 'candidate-length';
    length.textContent = `${game.data.source.totalTurns}수`;
    button.append(number, title, score, length);
    button.addEventListener('click', () => selectGame(game.id).catch(showError));
    list.append(button);
  });
  await selectGame(new URLSearchParams(location.search).get('game'), false);
} catch (error) { showError(error); }

function showError(error) {
  pause();
  console.error(error);
  $('[data-game-title]').textContent = '기보를 불러오지 못했습니다.';
  hint.textContent = '페이지를 새로고침해 주세요.';
}
