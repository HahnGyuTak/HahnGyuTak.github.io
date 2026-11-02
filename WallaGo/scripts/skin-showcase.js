import { mountBoardReplay } from './board-replay.js';

const descriptions = new Map([
  ['theme1', '기본'], ['theme4', '크리스탈'], ['theme2', '호박·옥'],
  ['theme3', '오닉스 골드'], ['theme5', '화염석·빙정석'], ['theme6', '고려청자·조선백자'],
]);

export function observeSkinShowcase() {
  const section = document.querySelector('#skins');
  if (!section) return;
  let started = false;
  const observer = new IntersectionObserver((entries) => {
    if (!entries.some((entry) => entry.isIntersecting) || started) return;
    started = true;
    observer.disconnect();
    mountShowcase(section).catch((error) => {
      console.error('Wall바둑 테마 미리보기를 불러오지 못했습니다.', error);
      section.querySelector('[data-skin-board]').textContent = '테마를 불러오지 못했습니다. 새로고침해 주세요.';
    });
  }, { rootMargin: '600px' });
  observer.observe(section);
}

async function mountShowcase(section) {
  const response = await fetch('./assets/themes.json');
  if (!response.ok) throw new Error('Theme catalog could not load.');
  const themes = await response.json();
  const board = section.querySelector('[data-skin-board]');
  const replay = await mountBoardReplay(board, { themes, hideResult: true, firstPlayer: 'A' });
  replay.setStep(replay.state.totalSteps, { animate: false });
  const steps = [...section.querySelectorAll('[data-skin-step]')];
  const choices = section.querySelector('[data-skin-choices]');
  let selected = -1;
  let scheduled = false;

  function choose(index) {
    if (index === selected) return;
    selected = index;
    const theme = themes[index];
    replay.setTheme(theme.id);
    section.dataset.activeSkin = theme.id;
    section.querySelector('[data-skin-name]').textContent = descriptions.get(theme.id);
    section.querySelector('[data-skin-availability]').textContent = theme.available ? '현재 앱에서 선택 가능' : '추가 테마 미리보기';
    steps.forEach((step, i) => { step.dataset.active = String(i === index); });
    [...choices.children].forEach((button, i) => { button.setAttribute('aria-pressed', String(i === index)); });
  }

  function update() {
    scheduled = false;
    const bounds = section.getBoundingClientRect();
    if (bounds.top > innerHeight || bounds.bottom < 0) return;
    const mobile = matchMedia('(max-width: 700px)').matches;
    const panelBottom = section.querySelector('.skin-sticky').getBoundingClientRect().bottom;
    const focus = mobile ? Math.min(innerHeight - 40, panelBottom + (innerHeight - panelBottom) * .5) : innerHeight * .55;
    let closest = 0, distance = Infinity;
    steps.forEach((step, i) => {
      const rect = step.getBoundingClientRect();
      const delta = Math.abs(rect.top + rect.height / 2 - focus);
      if (delta < distance) { closest = i; distance = delta; }
    });
    choose(closest);
  }

  for (const [index, theme] of themes.entries()) {
    const button = document.createElement('button');
    button.type = 'button';
    button.setAttribute('aria-label', `${descriptions.get(theme.id)}${theme.available ? '' : ' · 미리보기'}`);
    button.setAttribute('aria-pressed', 'false');
    for (const owner of ['A', 'B']) {
      const image = document.createElement('img');
      image.src = theme.assets[`player${owner}`];
      image.alt = '';
      image.width = 24;
      image.height = 24;
      image.loading = 'lazy';
      button.append(image);
    }
    button.addEventListener('click', () => {
      choose(index);
      const panel = section.querySelector('.skin-sticky');
      const mobile = matchMedia('(max-width: 700px)').matches;
      const focus = mobile ? Math.min(innerHeight - 40, panel.getBoundingClientRect().height + (innerHeight - panel.getBoundingClientRect().height) * .5) : innerHeight * .55;
      const rect = steps[index].getBoundingClientRect();
      window.scrollTo({ top: window.scrollY + rect.top + rect.height / 2 - focus,
        behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth' });
    });
    choices.append(button);
  }
  choose(0);
  const schedule = () => {
    if (!scheduled) { scheduled = true; requestAnimationFrame(update); }
  };
  window.addEventListener('scroll', schedule, { passive: true });
  window.addEventListener('resize', schedule, { passive: true });
  update();
}
