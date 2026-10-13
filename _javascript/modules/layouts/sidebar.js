const ATTR_DISPLAY = 'sidebar-display';
const COLLAPSED = 'data-sidebar-collapsed';

export function initSidebar() {
  const sidebar = document.getElementById('sidebar');
  const trigger = document.getElementById('sidebar-trigger');
  const desktopTrigger = document.getElementById('sidebar-desktop-toggle');
  const mask = document.getElementById('mask');
  const desktop = window.matchMedia('(min-width: 992px)');
  const root = document.documentElement;
  const sync = () => {
    const mobileOpen = document.body.hasAttribute(ATTR_DISPLAY);
    const collapsed = root.hasAttribute(COLLAPSED);
    const visible = desktop.matches ? !collapsed : mobileOpen;
    sidebar.inert = !visible;
    sidebar.setAttribute('aria-hidden', String(!visible));
    trigger.setAttribute('aria-expanded', String(mobileOpen));
    desktopTrigger.setAttribute('aria-expanded', String(!collapsed));
    desktopTrigger.setAttribute('aria-label', collapsed ? '사이드바 펼치기' : '사이드바 접기');
    desktopTrigger.querySelector('i').className = 'fa-solid fa-angles-' + (collapsed ? 'right' : 'left');
  };
  const setMobile = (open) => {
    document.body.toggleAttribute(ATTR_DISPLAY, open);
    sidebar.classList.toggle('z-2', open);
    mask.classList.toggle('d-none', !open);
    sync();
  };
  trigger.addEventListener('click', () => setMobile(!document.body.hasAttribute(ATTR_DISPLAY)));
  mask.addEventListener('click', () => setMobile(false));
  desktopTrigger.addEventListener('click', () => {
    const collapsed = root.toggleAttribute(COLLAPSED);
    try { localStorage.setItem('sidebar-collapsed', String(collapsed)); } catch { /* Session only. */ }
    sync();
  });
  desktop.addEventListener('change', () => {
    setMobile(false);
    sync();
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && document.body.hasAttribute(ATTR_DISPLAY)) {
      setMobile(false);
      trigger.focus();
    }
  });
  sync();
}
