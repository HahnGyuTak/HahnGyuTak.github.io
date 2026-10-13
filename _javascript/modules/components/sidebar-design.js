/** Typing Animation / Interactive Grid Pattern adapted from Magic UI (MIT). */
export function initSidebarDesign() {
  const sidebar = document.getElementById('sidebar');
  if (!sidebar) return;
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const typing = sidebar.querySelector('.tagline-typing');
  if (typing && !reducedMotion.matches) {
    const text = [...typing.dataset.text];
    let started = false;
    let index = 0;
    let timer;
    typing.textContent = '';
    typing.classList.add('is-typing');
    const tick = () => {
      if (reducedMotion.matches) {
        typing.textContent = text.join('');
        typing.classList.remove('is-typing');
        return;
      }
      typing.textContent = text.slice(0, ++index).join('');
      if (index < text.length) timer = setTimeout(tick, 90);
      else typing.classList.remove('is-typing');
    };
    const observer = new IntersectionObserver((entries) => {
      if (started || !entries.some((entry) => entry.isIntersecting)) return;
      started = true;
      observer.disconnect();
      timer = setTimeout(tick, 300);
    });
    observer.observe(typing);
    reducedMotion.addEventListener('change', () => {
      if (!reducedMotion.matches) return;
      clearTimeout(timer);
      observer.disconnect();
      typing.textContent = text.join('');
      typing.classList.remove('is-typing');
    });
  }

  const grid = sidebar.querySelector('.sidebar-grid');
  if (!grid) return;
  const finePointer = window.matchMedia('(hover: hover) and (pointer: fine)');
  let frame = 0;
  let x = 0;
  let y = 0;
  sidebar.addEventListener('pointermove', (event) => {
    if (!finePointer.matches) return;
    // Keep the grid highlight from appearing as an extra hover spot in glass controls.
    if (event.target.closest('.sidebar-bottom, .sidebar-theme')) {
      cancelAnimationFrame(frame);
      frame = 0;
      grid.classList.remove('is-hovered');
      return;
    }
    const bounds = sidebar.getBoundingClientRect();
    x = Math.floor((event.clientX - bounds.left) / 28) * 28;
    y = Math.floor((event.clientY - bounds.top) / 28) * 28;
    if (frame) return;
    frame = requestAnimationFrame(() => {
      grid.style.setProperty('--grid-x', x + 'px');
      grid.style.setProperty('--grid-y', y + 'px');
      grid.classList.add('is-hovered');
      frame = 0;
    });
  });
  sidebar.addEventListener('pointerleave', () => {
    cancelAnimationFrame(frame);
    frame = 0;
    grid.classList.remove('is-hovered');
  });
}
