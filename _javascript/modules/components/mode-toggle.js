/** System / Custom selection shares Chirpy's Theme storage and notifications. */
export function modeWatcher() {
  if (!Theme.isToggleable) return;
  const control = document.querySelector('.sidebar-theme');
  const system = document.getElementById('theme-system');
  const custom = document.getElementById('theme-custom');
  const toggle = document.getElementById('mode-toggle');
  const panel = document.querySelector('.theme-custom-panel');
  if (!control || !toggle || !panel) return;

  const remember = (theme) => {
    try { localStorage.setItem('custom-theme', theme); } catch { /* Session only. */ }
  };
  const sync = () => {
    const isCustom = !Theme.isSystemTheme;
    const dark = document.documentElement.dataset.bsTheme === Theme.Mode.DARK;
    system.checked = !isCustom;
    custom.checked = isCustom;
    toggle.checked = dark;
    toggle.setAttribute('aria-label', dark ? '라이트 모드로 전환' : '다크 모드로 전환');
    control.dataset.custom = String(isCustom);
    panel.inert = !isCustom;
    panel.setAttribute('aria-hidden', String(!isCustom));
    if (isCustom) remember(Theme.resolvedTheme);
  };
  system.addEventListener('change', () => {
    if (!system.checked) return;
    Theme.update(Theme.Mode.SYSTEM);
    sync();
  });
  custom.addEventListener('change', () => {
    if (!custom.checked) return;
    let preference;
    try { preference = localStorage.getItem('custom-theme'); } catch { /* Use current theme. */ }
    Theme.update([Theme.Mode.LIGHT, Theme.Mode.DARK].includes(preference) ? preference : Theme.resolvedTheme);
    sync();
  });
  toggle.addEventListener('change', () => {
    Theme.update(toggle.checked ? Theme.Mode.DARK : Theme.Mode.LIGHT);
    sync();
  });
  new MutationObserver(sync).observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['data-bs-theme', 'data-theme-persisted']
  });
  sync();
}
