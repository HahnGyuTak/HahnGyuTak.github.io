/** Ghost Jam by Vanessa Victorino (MIT code): assets/licenses/ui-components.txt. */
export function initSidebarGhost(sidebar) {
  const region = sidebar.querySelector('.sidebar-ghost-region');
  if (!region) return;
  const image = region.querySelector('.ghost-dancer');
  const fallback = region.querySelector('.ghost-fallback');
  const audio = region.querySelector('.ghost-audio');
  const button = region.querySelector('.ghost-play');
  const icon = button.querySelector('i');
  const state = region.querySelector('.ghost-state');
  const time = region.querySelector('.ghost-time');
  const progress = region.querySelector('.ghost-progress');
  const fill = region.querySelector('.ghost-progress-fill');
  const root = document.documentElement;
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  let frame = 0;
  let requested = false;
  let generation = 0;
  const visible = () => root.dataset.sidebarBackground === 'ghost' &&
    !document.hidden && sidebar.getAttribute('aria-hidden') !== 'true';
  const formatTime = (seconds) => {
    const value = Math.max(0, Math.floor(seconds || 0));
    return Math.floor(value / 60) + ':' + String(value % 60).padStart(2, '0');
  };
  function updateProgress() {
    const duration = Number.isFinite(audio.duration) && audio.duration > 0 ? audio.duration : 0;
    const percent = duration ? Math.max(0, Math.min(100, audio.currentTime / duration * 100)) : 0;
    const text = formatTime(audio.currentTime) + ' / ' + (duration ? formatTime(duration) : '--:--');
    time.textContent = text;
    fill.style.width = percent + '%';
    progress.setAttribute('aria-valuenow', String(Math.round(percent)));
    progress.setAttribute('aria-valuetext', text);
  }
  function showState(label, playing = false) {
    state.textContent = label;
    region.classList.toggle('is-playing', playing);
    button.setAttribute('aria-pressed', String(requested));
    const description = requested ? 'Ghost Jam 일시정지' : 'Ghost Jam 재생';
    button.setAttribute('aria-label', description);
    button.title = description;
    icon.className = requested ? 'fas fa-pause' : 'fas fa-play';
  }
  function animate() {
    frame = 0;
    updateProgress();
    if (requested && !audio.paused && visible()) frame = requestAnimationFrame(animate);
  }
  function pause() {
    generation++;
    requested = false;
    audio.pause();
    cancelAnimationFrame(frame);
    frame = 0;
    showState(audio.currentTime > 0 ? 'PAUSED' : 'READY');
    updateProgress();
  }
  function sync() {
    if (!visible()) pause();
    if (visible() && !reduced.matches) {
      if (image.getAttribute('src') !== image.dataset.src) image.src = image.dataset.src;
    } else {
      image.removeAttribute('src');
      image.hidden = true;
      fallback.hidden = false;
    }
  }
  image.addEventListener('load', () => {
    if (!visible() || reduced.matches || image.getAttribute('src') !== image.dataset.src) return;
    image.hidden = false;
    fallback.hidden = true;
  });
  image.addEventListener('error', () => {
    image.hidden = true;
    fallback.hidden = false;
  });
  button.addEventListener('click', async () => {
    if (requested) { pause(); return; }
    if (!visible()) return;
    const attempt = ++generation;
    requested = true;
    if (!audio.hasAttribute('src')) {
      audio.src = audio.dataset.src;
      audio.volume = 0.25;
    }
    showState('LOADING');
    try {
      await audio.play();
      if (!requested || !visible()) audio.pause();
    } catch (error) {
      if (attempt !== generation) return;
      requested = false;
      showState(error.name === 'AbortError' ? 'READY' : 'UNAVAILABLE');
    }
  });
  audio.addEventListener('playing', () => {
    if (!requested || !visible()) { pause(); return; }
    showState('JAMMING', true);
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(animate);
  });
  audio.addEventListener('waiting', () => {
    if (requested) showState('LOADING');
  });
  audio.addEventListener('pause', () => {
    requested = false;
    cancelAnimationFrame(frame);
    frame = 0;
    showState(audio.currentTime > 0 ? 'PAUSED' : 'READY');
    updateProgress();
  });
  audio.addEventListener('error', () => {
    requested = false;
    cancelAnimationFrame(frame);
    frame = 0;
    showState('UNAVAILABLE');
  });
  audio.addEventListener('loadedmetadata', updateProgress);
  audio.addEventListener('timeupdate', updateProgress);
  new MutationObserver(sync).observe(root, { attributes: true, attributeFilter: ['data-sidebar-background'] });
  new MutationObserver(sync).observe(sidebar, { attributes: true, attributeFilter: ['aria-hidden'] });
  document.addEventListener('visibilitychange', sync);
  reduced.addEventListener('change', sync);
  sync();
}
