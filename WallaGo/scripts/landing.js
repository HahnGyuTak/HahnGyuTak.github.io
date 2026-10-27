const clamp = (value, minimum, maximum) => Math.min(maximum, Math.max(minimum, value));
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const replayRoot = document.querySelector('[data-replay-board]');
const story = document.querySelector('#replay-story');
const progressInput = document.querySelector('#replay-progress');
const countOutput = document.querySelector('[data-replay-count]');
const hint = document.querySelector('[data-replay-hint]');
let replay;
let currentStep = -1;
let scrollFrame = 0;

function setReplayStep(step) {
  if (!replay) return;
  const { totalSteps, endTurn } = replay.state;
  const nextStep = clamp(Math.round(step), 0, totalSteps);
  if (currentStep === nextStep) return;
  currentStep = nextStep;
  replay.setStep(nextStep);
  progressInput.value = String(nextStep);
  const turnNumber = replay.state.frame.turnNumber;
  progressInput.setAttribute('aria-valuetext', `${endTurn}수 중 ${turnNumber}수`);
  countOutput.textContent = `${turnNumber} / ${endTurn}수`;
  hint.textContent = nextStep === totalSteps ? '한 판의 끝. 이제 직접 플레이해보세요.' : '스크롤하며 다음 수를 확인하세요.';
  document.querySelectorAll('[data-story-beat]').forEach((beat) => {
    beat.dataset.active = String(Number(beat.dataset.storyBeat) === Math.min(2, Math.floor(nextStep / 3)));
  });
}

function updateScrollReplay() {
  scrollFrame = 0;
  if (!story) return;
  const mobile = window.matchMedia('(max-width: 700px)').matches;
  const start = story.getBoundingClientRect().top + window.scrollY - window.innerHeight * (mobile ? 0.8 : 0.66);
  const range = Math.max(1, story.offsetHeight - window.innerHeight * (mobile ? 0.74 : 0.63));
  const progress = clamp((window.scrollY - start) / range, 0, 1);
  const totalSteps = replay?.state.totalSteps ?? Number(progressInput.max);
  setReplayStep(Math.floor(progress * (totalSteps + 0.999)));
}

function requestScrollUpdate() {
  if (!scrollFrame) scrollFrame = requestAnimationFrame(updateScrollReplay);
}

function startReplay() {
  setReplayStep(0);
  const mobile = window.matchMedia('(max-width: 700px)').matches;
  const target = story.getBoundingClientRect().top + window.scrollY - window.innerHeight * (mobile ? 0.8 : 0.66);
  window.scrollTo({ top: Math.max(0, target), behavior: reducedMotion.matches ? 'instant' : 'smooth' });
}

progressInput.addEventListener('input', (event) => setReplayStep(Number(event.target.value)));
window.addEventListener('scroll', requestScrollUpdate, { passive: true });
window.addEventListener('resize', requestScrollUpdate, { passive: true });
document.querySelector('[data-replay-restart]').addEventListener('click', startReplay);
document.querySelectorAll('a[href="#replay-story"]').forEach((link) => {
  link.addEventListener('click', (event) => {
    event.preventDefault();
    startReplay();
  });
});


const dialog = document.querySelector('[data-video-dialog]');
const player = document.querySelector('[data-video-player]');
const videoTrigger = document.querySelector('[data-video-open]');

function closeVideo() {
  dialog.close();
}

videoTrigger.addEventListener('click', () => {
  const iframe = document.createElement('iframe');
  iframe.src = 'https://www.youtube.com/embed/pIPlZ2D5rXQ?autoplay=1&playsinline=1&rel=0';
  iframe.title = '데블스 플랜: 데스룸 · Netflix Korea 공식 영상';
  iframe.allow = 'autoplay; encrypted-media; picture-in-picture; fullscreen';
  iframe.referrerPolicy = 'strict-origin-when-cross-origin';
  iframe.allowFullscreen = true;
  player.replaceChildren(iframe);
  document.body.classList.add('video-open');
  dialog.showModal();
});
document.querySelector('[data-video-close]').addEventListener('click', closeVideo);
dialog.addEventListener('click', (event) => {
  if (event.target !== dialog) return;
  const bounds = dialog.getBoundingClientRect();
  if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) closeVideo();
});
dialog.addEventListener('close', () => {
  player.replaceChildren();
  document.body.classList.remove('video-open');
  videoTrigger.focus({ preventScroll: true });
});
document.querySelector('[data-year]').textContent = String(new Date().getFullYear());

try {
  const { mountBoardReplay } = await import('./board-replay.js');
  replay = await mountBoardReplay(replayRoot, { dataUrl: './assets/replay.json', assetBase: './assets/game/' });
  progressInput.max = String(replay.state.totalSteps);
  progressInput.disabled = false;
  currentStep = -1;
  updateScrollReplay();
} catch (error) {
  console.error('Wall바둑 대국을 불러오지 못했습니다.', error);
  const message = document.createElement('p');
  message.className = 'replay-loading';
  message.textContent = '대국을 불러오지 못했습니다. 새로고침하면 다시 시도합니다.';
  replayRoot.replaceChildren(message);
}
