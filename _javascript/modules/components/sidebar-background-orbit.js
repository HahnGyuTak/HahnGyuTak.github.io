/** Rotate the background choices around the stationary profile photo. */
export function initSidebarBackgroundOrbit(sidebar) {
  const picker = sidebar.querySelector('.sidebar-background-picker');
  if (!picker) return;
  const root = document.documentElement;
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let rotation = parseFloat(getComputedStyle(picker).getPropertyValue('--sidebar-background-rotation')) || 0;
  let drag = null;
  let blockClick = false;
  let frame = 0;
  const angleAt = (event) => Math.atan2(event.clientY - drag.cy, event.clientX - drag.cx) * 180 / Math.PI;
  const normalize = (angle) => (angle % 360 + 360) % 360;
  const paint = () => root.style.setProperty('--sidebar-background-rotation', rotation + 'deg');
  const save = () => {
    try { localStorage.setItem('sidebar-background-rotation', String(rotation)); } catch { /* Session only. */ }
  };

  function stopMomentum() {
    cancelAnimationFrame(frame);
    frame = 0;
    picker.classList.remove('is-spinning');
    save();
  }

  function startMomentum(velocity) {
    // Degrees per millisecond. Exponential friction is independent of frame rate.
    const frictionTime = 320;
    let previousTime = performance.now();
    picker.classList.add('is-spinning');
    function coast(now) {
      frame = 0;
      const elapsed = Math.max(0, now - previousTime);
      previousTime = now;
      const decay = Math.exp(-elapsed / frictionTime);
      rotation = normalize(rotation + velocity * frictionTime * (1 - decay));
      velocity *= decay;
      paint();
      if (Math.abs(velocity) > 0.01) frame = requestAnimationFrame(coast);
      else stopMomentum();
    }
    frame = requestAnimationFrame(coast);
  }

  function finish(event, coast = false) {
    if (!drag || (event?.pointerId !== undefined && event.pointerId !== drag.id)) return;
    const { id, moved, velocity, time } = drag;
    drag = null;
    picker.classList.remove('is-rotating');
    if (picker.hasPointerCapture(id)) picker.releasePointerCapture(id);
    if (!moved) return;
    blockClick = true;
    save();
    // Holding still before release should not throw the wheel at an old speed.
    const idle = Math.max(0, (event?.timeStamp ?? performance.now()) - time);
    const releaseVelocity = velocity * Math.exp(-Math.max(0, idle - 32) / 50);
    if (coast && !reducedMotion.matches && idle < 150 && Math.abs(releaseVelocity) > 0.025) {
      startMomentum(releaseVelocity);
    }
  }

  function halt() {
    finish();
    stopMomentum();
  }

  picker.addEventListener('pointerdown', (event) => {
    if (!event.isPrimary || event.button !== 0 || drag) return;
    if (!event.target.closest('label, .orbit-track-hit')) return;
    stopMomentum();
    const bounds = picker.getBoundingClientRect();
    blockClick = false;
    drag = {
      id: event.pointerId, x: event.clientX, y: event.clientY,
      cx: bounds.left + bounds.width / 2, cy: bounds.top + bounds.height / 2,
      moved: false, angle: 0, velocity: 0, time: event.timeStamp
    };
    drag.angle = angleAt(event);
  });

  window.addEventListener('pointermove', (event) => {
    if (!drag || event.pointerId !== drag.id) return;
    if (!drag.moved) {
      if (Math.hypot(event.clientX - drag.x, event.clientY - drag.y) < 6) return;
      drag.moved = true;
      picker.classList.add('is-rotating');
      picker.setPointerCapture(drag.id);
    }
    event.preventDefault();
    const angle = angleAt(event);
    const delta = normalize(angle - drag.angle + 180) - 180;
    const elapsed = event.timeStamp - drag.time;
    if (elapsed > 0) {
      const speed = Math.max(-1.2, Math.min(1.2, delta / elapsed));
      const weight = 1 - Math.exp(-elapsed / 40);
      drag.velocity += (speed - drag.velocity) * weight;
    }
    drag.time = event.timeStamp;
    drag.angle = angle;
    rotation = normalize(rotation + delta);
    paint();
  });
  window.addEventListener('pointerup', (event) => finish(event, true));
  window.addEventListener('pointercancel', finish);
  picker.addEventListener('lostpointercapture', (event) => {
    if (event.target === picker) finish(event);
  });
  window.addEventListener('blur', halt);
  window.addEventListener('pagehide', halt);
  picker.addEventListener('keydown', stopMomentum);
  document.addEventListener('visibilitychange', () => { if (document.hidden) halt(); });
  reducedMotion.addEventListener('change', () => { if (reducedMotion.matches) halt(); });
  new MutationObserver(() => {
    if (sidebar.getAttribute('aria-hidden') === 'true') halt();
  }).observe(sidebar, { attributes: true, attributeFilter: ['aria-hidden'] });

  // A completed drag must not also select the button underneath its release point.
  picker.addEventListener('click', (event) => {
    if (!blockClick || event.detail === 0) return;
    blockClick = false;
    event.preventDefault();
    event.stopImmediatePropagation();
  }, true);
}
