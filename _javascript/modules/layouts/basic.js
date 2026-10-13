import { back2top, loadTooltip, modeWatcher } from '../components';
import { initReviewDesign } from '../components/review-design';
import { initSidebarDesign } from '../components/sidebar-design';

export function basic() {
  modeWatcher();
  back2top();
  loadTooltip();
  initReviewDesign();
  initSidebarDesign();
}
