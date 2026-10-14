/** Load Temple's WebGL pool only when its sidebar background is visible. */
export function initSidebarPool(sidebar) {
  const region = sidebar.querySelector('.sidebar-pool-region');
  const canvas = region?.querySelector('.sidebar-pool-canvas');
  if (!canvas) return;
  const root = document.documentElement;
  let factory, loading, instance;
  let failed = false;
  const visible = () => root.dataset.sidebarBackground === 'pool' &&
    !document.hidden && sidebar.getAttribute('aria-hidden') !== 'true';
  const fallback = (error) => {
    failed = true;
    canvas.hidden = true;
    canvas.removeAttribute('aria-busy');
    region.setAttribute('role', 'img');
    region.setAttribute('aria-label', '정적 수영장 배경');
    console.warn('Interactive pool is unavailable; showing its static background.', error);
  };
  function sync() {
    if (failed) return;
    if (instance) {
      instance.setActive(visible());
      return;
    }
    if (!visible()) return;
    if (factory) {
      try {
        instance = factory(canvas);
        canvas.setAttribute('aria-busy', 'false');
        instance.setActive(visible());
      } catch (error) { fallback(error); }
      return;
    }
    if (loading) return;
    canvas.setAttribute('aria-busy', 'true');
    const moduleUrl = canvas.dataset.renderer;
    loading = import(moduleUrl).then((module) => {
      factory = module.createPoolWater;
      loading = null;
      sync();
    }).catch(fallback);
  }
  new MutationObserver(sync).observe(root, { attributes: true, attributeFilter: ['data-sidebar-background'] });
  new MutationObserver(sync).observe(sidebar, { attributes: true, attributeFilter: ['aria-hidden'] });
  new ResizeObserver(() => { if (instance && visible()) instance.resize(); }).observe(region);
  document.addEventListener('visibilitychange', sync);
  sync();
}
