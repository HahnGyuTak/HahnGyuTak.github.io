/**
 * TIDES - A Cinematic Canvas Ocean by Chathura Jayasanka (MIT).
 * https://codepen.io/Chathura-Jayasanka/pen/emBMYWJ
 * Attribution / license: assets/licenses/ui-components.txt.
 * Render the scene's middle strip; keep its sun and horizon below the menu.
 */
const PALETTES = [
  { t: 0, name: 'DAWN', skyTop: [38, 44, 86], skyHor: [247, 176, 128],
    sun: [255, 238, 206], glow: [255, 178, 120], wFar: [176, 150, 150],
    wNear: [34, 62, 84], foam: [255, 244, 234], sunH: 0.1, glit: 0.7, star: 0 },
  { t: 0.28, name: 'MORNING', skyTop: [64, 134, 206], skyHor: [188, 222, 236],
    sun: [255, 255, 246], glow: [255, 250, 224], wFar: [120, 186, 196],
    wNear: [20, 92, 114], foam: [255, 255, 255], sunH: 0.55, glit: 0.5, star: 0 },
  { t: 0.5, name: 'MIDDAY', skyTop: [58, 142, 214], skyHor: [176, 216, 230],
    sun: [255, 255, 248], glow: [255, 252, 232], wFar: [96, 178, 188],
    wNear: [16, 96, 120], foam: [255, 255, 255], sunH: 0.92, glit: 0.45, star: 0 },
  { t: 0.68, name: 'GOLDEN HOUR', skyTop: [74, 92, 156], skyHor: [255, 202, 120],
    sun: [255, 236, 194], glow: [255, 168, 92], wFar: [206, 164, 118],
    wNear: [34, 78, 98], foam: [255, 244, 228], sunH: 0.3, glit: 0.95, star: 0 },
  { t: 0.84, name: 'SUNSET', skyTop: [48, 38, 86], skyHor: [255, 108, 68],
    sun: [255, 206, 148], glow: [255, 92, 58], wFar: [188, 98, 84],
    wNear: [30, 42, 72], foam: [255, 222, 200], sunH: 0.06, glit: 1, star: 0.15 },
  { t: 1, name: 'MOONLIT', skyTop: [8, 12, 30], skyHor: [34, 44, 82],
    sun: [228, 234, 255], glow: [140, 164, 216], wFar: [28, 42, 76],
    wNear: [6, 16, 32], foam: [196, 208, 234], sunH: 0.55, glit: 0.55, star: 1 }
];
const lerp = (a, b, t) => a + (b - a) * t;
const lerpRGB = (a, b, t) => a.map((v, i) => lerp(v, b[i], t));
const rgb = (c, a = 1) => 'rgba(' + c.map(Math.round).join(',') + ',' + a + ')';

function paletteAt(t) {
  let i = 0;
  while (i < PALETTES.length - 1 && t > PALETTES[i + 1].t) i++;
  const a = PALETTES[i], b = PALETTES[Math.min(i + 1, PALETTES.length - 1)];
  const k = Math.max(0, Math.min(1, (t - a.t) / (b.t - a.t || 1)));
  const p = { name: k < 0.5 ? a.name : b.name };
  for (const key of ['skyTop', 'skyHor', 'sun', 'glow', 'wFar', 'wNear', 'foam']) {
    p[key] = lerpRGB(a[key], b[key], k);
  }
  for (const key of ['sunH', 'glit', 'star']) p[key] = lerp(a[key], b[key], k);
  return p;
}

export function initSidebarOcean(sidebar) {
  const canvas = sidebar.querySelector('.sidebar-ocean');
  const slider = sidebar.querySelector('#sidebar-ocean-time');
  const controls = sidebar.querySelector('.sidebar-ocean-controls');
  const lastMenu = sidebar.querySelector('.nav-item:last-child a');
  const bottom = sidebar.querySelector('.sidebar-bottom');
  if (!canvas || !slider || !controls || !lastMenu || !bottom) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const root = document.documentElement;
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  let width = 0, height = 0, sceneWidth = 0, cropLeft = 0;
  let horizon = 0, skyStart = 0, oceanHeight = 0;
  let frame = 0, lastFrame = 0, elapsed = 0, pointer = 0.5;
  let time = 0.6, palette = paletteAt(time);
  try {
    const stored = localStorage.getItem('sidebar-ocean-time');
    const value = Number(stored);
    if (stored !== null && Number.isFinite(value) && value >= 0 && value <= 1000) time = value / 1000;
  } catch { /* Use the original afternoon scene. */ }
  slider.value = String(Math.round(time * 1000));
  const stars = Array.from({ length: 140 }, () => ({
    x: Math.random(), y: Math.random() * 0.4,
    r: Math.random() * 1.2 + 0.3, tw: Math.random() * Math.PI * 2
  }));
  const clouds = Array.from({ length: 5 }, () => ({
    x: Math.random(), y: 0.08 + Math.random() * 0.18,
    w: 0.18 + Math.random() * 0.22, speed: 0.0009 + Math.random() * 0.0012
  }));
  const birds = Array.from({ length: 4 }, () => ({
    x: Math.random(), y: 0.15 + Math.random() * 0.18,
    speed: 0.0024 + Math.random() * 0.0024,
    size: 8 + Math.random() * 6, flap: Math.random() * Math.PI * 2
  }));

  const selected = () => root.dataset.sidebarBackground === 'ocean';
  const visible = () => selected() && !document.hidden && sidebar.getAttribute('aria-hidden') !== 'true';
  const position = (element) => element.getBoundingClientRect().top -
    sidebar.getBoundingClientRect().top + sidebar.scrollTop;
  function resize() {
    width = sidebar.clientWidth;
    height = Math.max(sidebar.clientHeight, position(bottom) + bottom.offsetHeight +
      (parseFloat(getComputedStyle(bottom).marginBottom) || 0));
    sceneWidth = Math.max(1024, width * 4);
    cropLeft = (sceneWidth - width) / 2;
    skyStart = position(lastMenu) + lastMenu.offsetHeight + 14;
    const skyEnd = Math.max(skyStart + 60, position(controls) - 18);
    horizon = skyStart + (skyEnd - skyStart) * 0.72;
    oceanHeight = Math.max(1, height - horizon);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.ceil(width * dpr);
    canvas.height = Math.ceil(height * dpr);
    canvas.style.height = height + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, -cropLeft * dpr, 0);
  }
  function updatePalette() {
    palette = paletteAt(time);
    const minutes = Math.round((5 + time * 18) * 60);
    const clock = String(Math.floor(minutes / 60)).padStart(2, '0') + ':' +
      String(minutes % 60).padStart(2, '0');
    slider.setAttribute('aria-valuetext', palette.name + ', ' + clock);
    slider.title = palette.name + ' · ' + clock;
  }
  function draw() {
    if (!width || !height) return;
    const p = palette;
    const right = cropLeft + width;
    const sunX = sceneWidth / 2 + (pointer - 0.5) * width * 0.3;
    const skySpan = horizon - skyStart;
    const sunY = horizon - p.sunH * skySpan * 0.82;
    const sunRadius = Math.min(28, Math.max(16, skySpan * 0.18));
    const sky = ctx.createLinearGradient(0, 0, 0, horizon + oceanHeight * 0.1);
    sky.addColorStop(0, rgb(p.skyTop));
    sky.addColorStop(0.7, rgb(lerpRGB(p.skyTop, p.skyHor, 0.55)));
    sky.addColorStop(1, rgb(p.skyHor));
    ctx.fillStyle = sky;
    ctx.fillRect(cropLeft, 0, width, horizon + 2);

    if (p.star > 0.01) {
      for (const s of stars) {
        ctx.fillStyle = rgb([255, 255, 255], p.star * (0.5 + 0.5 * Math.sin(elapsed * 2 + s.tw)) * 0.9);
        ctx.beginPath();
        ctx.arc(s.x * sceneWidth, s.y * horizon, s.r, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    const glow = ctx.createRadialGradient(sunX, sunY, 0, sunX, sunY, Math.min(sceneWidth, height) * 0.5);
    glow.addColorStop(0, rgb(p.glow, 0.55));
    glow.addColorStop(0.25, rgb(p.glow, 0.22));
    glow.addColorStop(1, rgb(p.glow, 0));
    ctx.fillStyle = glow;
    ctx.fillRect(cropLeft, 0, width, horizon + oceanHeight * 0.4);
    const disc = ctx.createRadialGradient(sunX, sunY, 0, sunX, sunY, sunRadius);
    disc.addColorStop(0, rgb(p.sun));
    disc.addColorStop(0.7, rgb(p.sun, 0.95));
    disc.addColorStop(1, rgb(p.sun, 0.2));
    ctx.fillStyle = disc;
    ctx.beginPath();
    ctx.arc(sunX, sunY, sunRadius, 0, Math.PI * 2);
    ctx.fill();

    for (const c of clouds) {
      const x = ((c.x + elapsed * c.speed + 0.3) % 1.6 - 0.3) * sceneWidth;
      const y = c.y * horizon, cw = c.w * sceneWidth;
      ctx.fillStyle = rgb(lerpRGB(p.skyHor, [255, 255, 255], 0.25), 0.16);
      for (let j = 0; j < 4; j++) {
        ctx.beginPath();
        ctx.ellipse(x + j * cw * 0.22, y + Math.sin(j) * 6,
          cw * (0.3 - j * 0.04), cw * 0.06, 0, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    for (const b of birds) {
      const x = ((b.x + elapsed * b.speed + 0.2) % 1.4 - 0.2) * sceneWidth;
      const y = b.y * horizon, wing = Math.sin(elapsed * 9 + b.flap) * b.size * 0.5;
      ctx.strokeStyle = rgb(lerpRGB(p.skyTop, [0, 0, 0], 0.3), 0.5);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x - b.size, y + wing);
      ctx.quadraticCurveTo(x, y - b.size * 0.3, x, y);
      ctx.quadraticCurveTo(x, y - b.size * 0.3, x + b.size, y + wing);
      ctx.stroke();
    }
    const haze = ctx.createLinearGradient(0, horizon - 40, 0, horizon + 40);
    haze.addColorStop(0, rgb(p.skyHor, 0));
    haze.addColorStop(0.5, rgb(p.skyHor, 0.45));
    haze.addColorStop(1, rgb(p.wFar, 0));
    ctx.fillStyle = haze;
    ctx.fillRect(cropLeft, horizon - 40, width, 80);

    for (let i = 0; i < 26; i++) {
      const depth = i / 25;
      const yTop = horizon + Math.pow(depth, 1.9) * oceanHeight;
      const amplitude = lerp(0.6, 30, depth), wavelength = lerp(46, 340, depth);
      const phase = elapsed * lerp(0.25, 0.9, depth) + i * 0.9;
      const color = lerpRGB(p.wFar, p.wNear, depth);
      const waveY = (x) => yTop + Math.sin(x / wavelength + phase) * amplitude +
        Math.sin(x / (wavelength * 0.4) + phase * 1.6) * amplitude * 0.3;
      ctx.beginPath();
      ctx.moveTo(cropLeft, height);
      ctx.lineTo(cropLeft, waveY(cropLeft));
      for (let x = cropLeft; x <= right + 6; x += 6) ctx.lineTo(x, waveY(x));
      ctx.lineTo(right, height);
      ctx.closePath();
      ctx.fillStyle = rgb(color);
      ctx.fill();
      ctx.lineWidth = lerp(0.6, 2.2, depth);
      ctx.beginPath();
      ctx.moveTo(cropLeft, waveY(cropLeft));
      for (let x = cropLeft; x <= right + 6; x += 6) ctx.lineTo(x, waveY(x));
      ctx.strokeStyle = rgb(lerpRGB(color, p.sun, 0.55), lerp(0.05, 0.3, depth));
      ctx.stroke();
      if (depth > 0.62) {
        for (let x = cropLeft; x <= right; x += 9) {
          if (Math.sin(x / wavelength + phase) > 0.55 && Math.random() > 0.45) {
            ctx.fillStyle = rgb(p.foam, (depth - 0.62) / 0.38 * (0.18 + Math.random() * 0.35));
            ctx.fillRect(x + (Math.random() - 0.5) * 6, waveY(x) - Math.random() * 3,
              1.5 + Math.random() * 3, 1.5 + Math.random() * 2);
          }
        }
      }
    }
    for (let i = 0; i < 220; i++) {
      const depth = Math.random(), spread = lerp(6, sceneWidth * 0.3, depth);
      const x = sunX + (Math.random() - 0.5) * 2 * spread;
      const fade = 1 - Math.min(1, Math.abs(x - sunX) / (spread + 1));
      const alpha = fade * fade * (0.25 + Math.random() * 0.75) * p.glit * (1 - depth * 0.25);
      if (alpha < 0.02 || x < cropLeft || x > right) continue;
      ctx.fillStyle = rgb(p.sun, alpha * 0.85);
      ctx.fillRect(x, horizon + Math.pow(depth, 1.5) * oceanHeight,
        1 + Math.random() * (2 + depth * 4), 1 + depth);
    }
    const vignette = ctx.createRadialGradient(sceneWidth / 2, horizon, height * 0.25,
      sceneWidth / 2, horizon, height * 0.9);
    vignette.addColorStop(0, 'rgba(0,0,0,0)');
    vignette.addColorStop(1, 'rgba(0,0,8,0.34)');
    ctx.fillStyle = vignette;
    ctx.fillRect(cropLeft, 0, width, height);
  }
  function animate(now) {
    frame = 0;
    if (!visible() || reduced.matches) return;
    if (now - lastFrame >= 1000 / 30) {
      elapsed += Math.min((now - lastFrame) / 1000, 0.1);
      lastFrame = now;
      draw();
    }
    frame = requestAnimationFrame(animate);
  }
  function sync() {
    cancelAnimationFrame(frame);
    frame = 0;
    slider.disabled = !selected();
    if (!visible()) return;
    resize();
    draw();
    lastFrame = performance.now();
    if (!reduced.matches) frame = requestAnimationFrame(animate);
  }
  slider.addEventListener('input', () => {
    time = Math.max(0, Math.min(1, Number(slider.value) / 1000));
    updatePalette();
    if (visible()) draw();
    try { localStorage.setItem('sidebar-ocean-time', slider.value); } catch { /* Session only. */ }
  });
  sidebar.addEventListener('pointermove', (event) => {
    if (!selected()) return;
    pointer = Math.max(0, Math.min(1, (event.clientX - sidebar.getBoundingClientRect().left) / sidebar.clientWidth));
    if (reduced.matches && visible()) draw();
  });
  const observer = new ResizeObserver(sync);
  for (const element of [sidebar, sidebar.querySelector('nav'), controls, bottom]) observer.observe(element);
  new MutationObserver(sync).observe(root, { attributes: true, attributeFilter: ['data-sidebar-background'] });
  new MutationObserver(sync).observe(sidebar, { attributes: true, attributeFilter: ['aria-hidden'] });
  document.addEventListener('visibilitychange', sync);
  reduced.addEventListener('change', sync);
  window.addEventListener('resize', sync);
  updatePalette();
  sync();
}
