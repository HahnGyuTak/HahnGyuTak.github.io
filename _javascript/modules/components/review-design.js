/** UI adaptations and licenses: assets/licenses/ui-components.txt. */
export function initReviewDesign() {
  const finePointer = window.matchMedia('(hover: hover) and (pointer: fine)');
  document.querySelectorAll('#post-list .magic-card').forEach((card) => {
    let frame = 0;
    let x = 0;
    let y = 0;
    card.addEventListener('pointermove', (event) => {
      if (!finePointer.matches) return;
      const bounds = card.getBoundingClientRect();
      x = event.clientX - bounds.left;
      y = event.clientY - bounds.top;
      if (frame) return;
      frame = requestAnimationFrame(() => {
        card.style.setProperty('--magic-x', `${x}px`);
        card.style.setProperty('--magic-y', `${y}px`);
        frame = 0;
      });
    });
  });

  const content = document.querySelector('article[data-paper-review] > .content');
  const progress = document.getElementById('review-progress');
  if (!content) return;
  // A transformed mobile main-wrapper must not move this viewport-wide bar.
  if (progress) document.body.appendChild(progress);
  // Keep nested explanations open with their parent; collapse other peers.
  content.addEventListener('toggle', (event) => {
    const detail = event.target;
    if (!(detail instanceof HTMLDetailsElement) || !detail.open) return;
    content.querySelectorAll('details[open]').forEach((other) => {
      if (other !== detail && !other.contains(detail) && !detail.contains(other)) other.open = false;
    });
  }, true);
  if (!progress) return;
  let pending = false;
  let previousPercent = -1;
  const update = () => {
    const bounds = content.getBoundingClientRect();
    const topbar = document.getElementById('topbar-wrapper');
    const offset = topbar ? topbar.getBoundingClientRect().height : 0;
    const length = bounds.height - window.innerHeight + offset;
    const ratio = length > 0
      ? Math.min(1, Math.max(0, (offset - bounds.top) / length))
      : (bounds.bottom <= window.innerHeight ? 1 : 0);
    progress.style.setProperty('--review-progress', ratio);
    const percent = Math.round(ratio * 100);
    if (percent !== previousPercent) {
      progress.setAttribute('aria-valuenow', String(percent));
      previousPercent = percent;
    }
    pending = false;
  };
  const schedule = () => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(update);
  };
  window.addEventListener('scroll', schedule, { passive: true });
  window.addEventListener('resize', schedule);
  new ResizeObserver(schedule).observe(content);
  update();
}
