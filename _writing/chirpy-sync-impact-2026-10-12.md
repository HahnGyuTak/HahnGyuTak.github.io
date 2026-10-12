# Chirpy 원본 동기화 영향 분석

작성일: 2026-10-12. 대상: `HahnGyuTak/HahnGyuTak.github.io`의 현재 `main`과 `cotes2020/jekyll-theme-chirpy`의 `master`.

**일반적인 Git 병합에서는 기존 글과 논문 이미지 파일이 보존되지만, 테마 화면·설정 형식·생성 파일·배포 절차가 바뀐다. 이번 비교에서는 충돌도 발생하므로, 원본을 동기화하는 버튼 한 번으로 업데이트를 마칠 수 있는 상태가 아니다.** 개인 설정과 사용자 코드를 이관하고 개발·production 환경에서 렌더링을 검증하는 작업이 필요하다.

## 1. 비교 기준과 확인 방법

| 기준 | 커밋 | 의미 |
| --- | --- | --- |
| 마지막 공통 조상 | `897b57bfb4404c9ccbb5909008b7029980272e65` | 2023-10-10의 `chore(release): 6.2.3` |
| 내 블로그 | `c5f35889aeb23474705c91092c0972e0d224a58b` | UMM 글과 작성 지침까지 반영한 현재 HEAD |
| 원본 master | `ae1057257b3927d6474deb549bdfafa71244e080` | 패키지 버전 7.6.0이며, 해당 릴리스 이후 master 변경도 포함 |

`207 commits ahead / 368 commits behind`는 위 공통 조상에서 갈라진 양쪽의 고유 커밋 수다. 원본에만 있는 368개와 내 저장소에만 있는 207개가 모두 확인됐다. 368개의 새 기능이라는 뜻은 아니다. 의존성 업데이트, 번역, 개발 도구, 버그 수정, 릴리스 커밋도 들어 있다.

이 공통 조상은 변경 대조에 필요한 기준이다. Git 기록만으로 GitHub에서 Fork 버튼을 누른 정확한 시각까지 확정하지는 않는다. 사용자가 말한 “fork 이후 달라진 부분”을 분석할 때는 이 기준에서 양쪽이 각각 바꾼 내용을 비교하면 된다.

실제 저장소는 병합하지 않았다. `/private/tmp/chirpy-sync-audit.qKeWhy/repo`에 별도 Git 복사본을 만들고 원본 주소에서 master를 가져왔다. 다음 비교를 수행했다.

1. 공통 조상 → 원본: 원본에서 바뀐 파일과 기능 확인.
2. 공통 조상 → 내 HEAD: 글, 이미지, 설정, 사용자 코드 확인.
3. `git merge-tree --write-tree HEAD chirpy/master`: 작업 디렉토리를 바꾸지 않고 양쪽을 병합했을 때의 충돌과 잠정 트리 계산.
4. 내 HEAD와 잠정 트리의 blob ID 대조: 글·이미지·작성 지침의 파일 내용 보존 여부 확인.

잠정 병합 트리는 `fe778fc539f31b6a6a07691070b00d8bdf77ce4c`다. **충돌 마커가 들어 있는 중간 결과이므로 실행 가능한 완성본이 아니다.** 이 분석에서 병합 후 사이트 빌드가 성공하거나 브라우저 화면이 정상임을 확인한 것으로 해석하면 안 된다.

## 2. 수치로 확인한 변경 범위

| 항목 | 결과 |
| --- | --- |
| 원본의 고유 커밋 | 368개 |
| 내 저장소의 고유 커밋 | 207개 |
| 원본의 변경 파일 항목, 이름 변경 감지 적용 | 227개: 추가 77, 수정 105, 삭제 33, 이름 변경 12 |
| 원본의 텍스트 변경량 | 8,029줄 추가 / 4,928줄 삭제 |
| 양쪽에서 모두 건드린 경로, 이름 변경 전후를 별도 계산 | 38개 |
| Git 충돌 | 32건, 관련 경로 34개 |
| 충돌 종류 | 수정/삭제 21건, 내용 충돌 10건, 이름 변경 충돌 1건 |

전체 227개 변경 항목은 [파일 목록 TSV](chirpy-upstream-files-2026-10-12.tsv)에 기록했다. `A/M/D/R`은 공통 조상 대비 원본의 추가/수정/삭제/이름 변경을 뜻한다. TSV의 “자동 병합”은 Git에서 충돌이 없다는 의미이며, 실행·렌더링 호환성 보증은 아니다. 삭제 항목의 `upstream_path`는 삭제 대상 경로를 나타낸다.

## 3. 기존 글과 자료는 어떻게 되는가

| 내 HEAD의 대상 | 잠정 병합 결과 |
| --- | --- |
| `_posts/`의 Markdown 글 44편 | 모두 같은 blob ID: 본문과 front matter 보존 |
| `_posts/`의 나머지 추적 파일 8개 | 모두 보존: `.DS_Store` 6개와 보조 Python/YAML 파일 2개 |
| favicon을 제외한 `assets/img/`의 추적 파일 394개 | 모두 같은 blob ID: 이미지 389개와 `.DS_Store` 5개 |
| `_writing/`의 기존 문서 5개와 `AGENTS.md` | 모두 보존 |

총 452개 보호 대상의 변경·삭제는 없었다. UMM Markdown과 해당 논문 figure/table도 포함된다. 주제별 디렉토리, categories/tags, `$`·`$$` 수식, HTML 토글 같은 글 작성 형식 자체는 원본 테마에서 덮어쓰는 대상이 아니었다.

단, **파일 보존과 화면 보존은 별개다.** 같은 Markdown도 CSS, 레이아웃, 수식 엔진, 이미지 URL 생성기가 바뀌면 화면에서 다르게 보이거나 깨질 수 있다. 원본 설정으로 파일 전체를 교체하거나 강제로 원본 상태에 맞추는 조작에 대해서까지 위 보존 결과가 적용되는 것은 아니다.

현재 작업 디렉토리의 수정된 `.DS_Store`, `assets/img/.DS_Store` 및 미추적 `assets/img/무제 2/`는 커밋 기준 비교에 포함하지 않았고 수정하지 않았다. “44편”은 실제 `.md` 수이며, `_posts/`의 모든 추적 파일 수인 52개와 구분한다.

## 4. 이용자가 체감하는 테마 변화

다음은 원본 변경 기록과 실제 템플릿·JavaScript·SCSS를 대조한 결과다. 정상 이관과 생성 파일 재빌드가 완료됐을 때 적용되는 변화이며, 충돌 상태 그대로의 화면을 의미하지 않는다.

| 영역 | 원본의 변화 | 이 블로그에서의 영향 |
| --- | --- | --- |
| 목차 | 모바일 팝업 목차와 데스크톱 목차 스크롤 추가 | 긴 논문 글을 모바일에서 탐색하기 편해짐. 새 목차 DOM과 새 JS를 함께 적용해야 함 |
| 본문과 화면 | SCSS 구조 재편, 인용·코드·제목·사이드바·명암 모드 및 반응형 처리 개선 | 글 문장은 그대로여도 색, 간격, 줄 배치 등이 바뀜. 서지정보 quote, 표, 긴 수식을 실제 글로 대조해야 함 |
| 그림 확대 | Magnific Popup에서 GLightbox로 교체, 이미지 클래스·캡션 처리 개선 | 논문 figure 확대 화면과 조작 방식이 달라짐. 기존 이미지 크기·정렬 클래스 검증 필요 |
| 검색 | JSON 특수문자 escaping, 문단 사이 공백 및 description이 있는 글의 전문 검색 수정 | 긴 한국어 논문 리뷰 검색의 안정성 개선 |
| 최근 글 목록 | “Recently Updated”에 최근 발행 글도 포함 | 오른쪽 목록에 나타나는 글 구성이 달라질 수 있음 |
| 수식 | equation numbering 설정, 별도 MathJax 설정 파일, 새로고침 시 로딩 문제 수정 | 새 테마 경로를 사용하도록 이관한 경우 효과가 적용됨. 현재 사용자 로딩 코드와의 정리가 필요 |
| 명암 모드 | 사용자 선택 저장과 관련 동작 개선 | 테마 선택·재방문 동작 변경. 새 `theme.min.js` 필요 |
| PWA | 캐시 제외 경로 설정, 알림 UI, iOS 표시와 캐시 동작 개선 | 현재 `pwa.enabled: true`이므로 이관하면 영향을 받음. production 검증 필요 |
| 선택 기능 | 글 편집 링크, 오디오/비디오/Spotify embed, 분석 서비스·언어·소셜 옵션 추가 | 코드는 추가되지만 필요한 설정과 사용 없이 모든 기능이 화면에 자동 표시되지는 않음. 글 편집 링크는 기본 비활성화 |

기능·수정 근거: [원본 변경 기록](https://github.com/cotes2020/jekyll-theme-chirpy/blob/ae1057257b3927d6474deb549bdfafa71244e080/docs/CHANGELOG.md), [글 레이아웃](https://github.com/cotes2020/jekyll-theme-chirpy/blob/ae1057257b3927d6474deb549bdfafa71244e080/_layouts/post.html), [SCSS](https://github.com/cotes2020/jekyll-theme-chirpy/tree/ae1057257b3927d6474deb549bdfafa71244e080/_sass).

## 5. 이 저장소에서 반드시 이관할 설정과 사용자 코드

### 5.1 설정 이름 변경

| 기존 설정 | 새 설정 | 현재 저장소에 적용할 때 |
| --- | --- | --- |
| `google_analytics.id` | `analytics.google.id` | 기존 측정 ID를 이관. default 레이아웃의 직접 삽입 gtag와 중복 실행되지 않도록 하나로 정리 |
| `google_site_verification` | `webmaster_verifications.google` | 현재 검증 값을 새 위치에 보존 |
| `comments.active` | `comments.provider` | 현재 설정값은 비어 있지만 post 레이아웃에서 Utterances를 직접 로딩하고 있음. 설정만 복사하는 것으로는 현재 댓글 동작을 설명할 수 없음 |
| `img_cdn` | `cdn` | 현재 값은 비어 있음. 원본의 데모 CDN 값으로 바꾸면 로컬 그림 경로가 데모 CDN으로 연결될 수 있음 |
| 글의 `img_path` | `media_subpath` | 현재 44편에서 이 기존 키는 발견되지 않았으므로 일괄 수정할 대상은 없었음 |
| `pwa.enabled` | 기존 키 유지 + `pwa.cache.enabled`, `deny_paths` | 현재 활성화 상태를 유지하면서 새 캐시 정책을 정해야 함 |

`title: ImTak`, 블로그 URL, `Asia/Seoul`, 작성자·SNS·avatar·plugins 등 개인 값은 유지할 대상이다. 일반 병합이 이 값을 전부 원본 값으로 재설정하는 것은 아니다. `_config.yml`의 충돌을 해결할 때 원본 파일 전체를 선택하면 개인 값까지 바뀔 수 있으므로 새 구조에 기존 값을 이관해야 한다.

근거: [새 설정](https://github.com/cotes2020/jekyll-theme-chirpy/blob/ae1057257b3927d6474deb549bdfafa71244e080/_config.yml), [미디어 URL 생성기](https://github.com/cotes2020/jekyll-theme-chirpy/blob/ae1057257b3927d6474deb549bdfafa71244e080/_includes/media-url.html).

### 5.2 수식: 글 수정 없이 로딩 경로를 맞출 수 있지만 검증이 필요

현재 44편 모두 `use_math` 키를 사용한다. 새 테마의 기본 로더는 `page.math`를 조건으로 사용한다. 그러나 잠정 병합의 `_includes/head.html`에는 기존 `page.use_math` → `mathjax_support.html` 경로가 보존됐다. 따라서 “동기화하면 모든 수식이 바로 사라진다”는 결론은 근거가 없다.

실제 주의점은 기존 코드에 MathJax 2 계열, KaTeX 0.12.0, 여러 레이아웃에서의 수식 로딩이 함께 있다는 것이다. 새 테마 기본 경로까지 추가로 켜면 중복 처리할 여지가 있다. 이관 시 `use_math`를 지원하는 조건을 유지하고 한 수식 로더로 정리하는 방식이 가능하다. 본문의 `$`·`$$` 구분자와 기호 bullet 설명 형식은 그대로 유지할 수 있다. 새 MathJax 설정도 dollar 구분자를 지원한다.

기존 식이 새 엔진에서 모두 동일하게 보이는지는 정적 비교로 확정하지 않았다. 긴 수식, `aligned`, 번호·참조, HTML 토글 안 수식을 대표 글에서 확인해야 한다.

근거: [새 JS 선택기](https://github.com/cotes2020/jekyll-theme-chirpy/blob/ae1057257b3927d6474deb549bdfafa71244e080/_includes/js-selector.html), [새 MathJax 설정](https://github.com/cotes2020/jekyll-theme-chirpy/blob/ae1057257b3927d6474deb549bdfafa71244e080/assets/js/data/mathjax.js).

### 5.3 메뉴·댓글·하단 문구

| 기존 사용자 변경 | 잠정 병합에서의 상태 | 이관 판단 |
| --- | --- | --- |
| Sidebar의 About 목적지를 `https://hahngt.github.io`로 변경 | `_includes/sidebar.html` 충돌 | 새 마크업에 기존 외부 링크 동작 보존 |
| Twitter 대신 Instagram 아이콘/사용자명 처리 | contact 데이터는 자동 병합, sidebar는 충돌 | 아이콘 데이터만 보존해도 주소 생성 코드가 사라지면 링크가 달라짐. 양쪽 보존 |
| Footer의 Jekyll/Chirpy 표시를 주석 처리 | 자동 병합에서도 주석 유지 | 원본에 버전 표시가 생겼다고 현재 블로그에 테마 문구가 자동 재등장하는 것은 아님 |
| `en.yml`의 라이선스 template와 copyright brief를 비움 | 자동 병합에서 유지 | 기존 의도 보존 |
| post 레이아웃에 직접 삽입한 BlogPosting JSON-LD | 레이아웃 상단 충돌 | SEO 플러그인 출력과 함께 중복 여부 검토 |
| post 레이아웃 하단의 직접 Utterances 코드 | 잠정 병합에서 유지 | 새 `comments.provider` 방식과 함께 켜서 댓글이 두 번 생성되지 않도록 정리 |
| head에 추가한 수식 hook·KaTeX·RSS 링크 | 자동 병합에서 유지 | 자동 병합됐더라도 새 기본 기능과 겹치는 부분 검토 |

### 5.4 Favicon은 기존 파일을 남기는 것만으로 충분하지 않음

기존 favicon 관련 10개 경로에서 충돌한다. 추가로 원본의 `favicon.svg`, `favicon-96x96.png`, `web-app-manifest-192x192.png`, `web-app-manifest-512x512.png`가 충돌 없이 들어온다. 새 `_includes/favicons.html`은 SVG/96px 파일을 참조하고 새 manifest는 `web-app-manifest-*`를 사용한다.

기존 `favicon.ico`만 보존하면 브라우저가 새 원본 SVG를 선택하거나 설치 아이콘이 원본 것으로 보일 수 있다. **기존 블로그 아이콘을 새 파일 이름·크기와 manifest 참조에 맞춰 보존해야 한다.** 이는 논문 이미지 경로 394개 보존 결과와 별도다.

### 5.5 RSS 생성 방식

사용자는 테마의 `assets/feed.xml`을 삭제하고 `jekyll-feed` 플러그인을 추가했다. 원본은 해당 파일을 수정했으므로 수정/삭제 충돌로 다시 나타난다. 그 파일은 `/feed.xml`로 출력하도록 되어 있다. 무조건 되살리면 현재 플러그인의 feed 생성 방식과 겹칠 수 있으므로, 기존 삭제 의도를 유지할지 또는 feed 구현을 변경할지 명시적으로 정리해야 한다.

## 6. 충돌 해결만으로는 부족한 빌드 변화

### 6.1 새 생성 파일이 잠정 병합 트리에 없음

| 새 코드가 요구하는 파일 | 참조/생성 위치 | 잠정 병합 트리 |
| --- | --- | --- |
| `_sass/vendors/_bootstrap.scss` | production CSS가 `main.bundle.scss`에서 import. `purgecss.js`가 생성 | 없음 |
| `assets/js/dist/theme.min.js` | head에서 항상 로딩. Rollup이 생성 | 없음 |
| `assets/js/dist/app.min.js` | front matter를 통해 `/app.min.js`로 출력. production PWA 로더가 참조 | 없음 |
| `assets/js/dist/sw.min.js` | front matter를 통해 `/sw.min.js`로 출력. 새 PWA 경로 | 없음 |

현재 저장소에는 예전 `categories/commons/home/misc/page/post.min.js` 6개만 추적되어 있다. 원본 master는 새 생성물을 Git에 포함하지 않으므로 일반 병합도 그 파일을 만들어 주지 않는다. 기존 6개도 새 JS 소스에 맞춰 재생성해야 한다.

현재 배포 workflow는 Ruby 설정 후 곧바로 `JEKYLL_ENV=production bundle exec jekyll b`를 실행하며 Node/CSS/JS 빌드 단계가 없다. production CSS의 import 대상도 없으므로, **현재 배포 절차를 그대로 쓰는 원본 병합은 필요한 생성물을 공급하지 못한다.** 이는 소스 의존성과 파일 존재 여부를 확인한 결과이며, 충돌을 해결한 완성본을 실제 빌드해 얻은 오류 로그는 아니다.

이관할 때는 새 의존성을 설치하고 `npm run build`를 실행하는 절차를 마련해야 한다. 출력물을 저장소에 포함하는 방식과 배포 시 생성하는 방식 중 일관된 방식을 적용하면 된다. 새 `.gitignore`도 출력물을 무시하므로 함께 조정해야 한다. 기본 개발 서버에서는 Bootstrap을 CDN으로 불러오는 반면 production에서는 위 bundle을 요구하므로 **`bundle exec jekyll serve` 확인만으로 GitHub 배포 성공을 보장하지 못한다.**

근거: [CSS 진입점](https://github.com/cotes2020/jekyll-theme-chirpy/blob/ae1057257b3927d6474deb549bdfafa71244e080/assets/css/jekyll-theme-chirpy.scss), [CSS 생성](https://github.com/cotes2020/jekyll-theme-chirpy/blob/ae1057257b3927d6474deb549bdfafa71244e080/purgecss.js), [JS 생성](https://github.com/cotes2020/jekyll-theme-chirpy/blob/ae1057257b3927d6474deb549bdfafa71244e080/rollup.config.js).

### 6.2 Ruby·Gem과 프런트엔드 의존성

- 현재 workflow의 Ruby 3.2.2는 원본 gemspec의 `~> 3.1` 요구 범위에 들어간다. 원본 배포 예제가 Ruby 3.4로 바뀌었다고 해서 현재 3.2.2 자체가 즉시 호환 불가라는 뜻은 아니다.
- 로컬 lockfile은 이미 `jekyll-sass-converter 3.0.0`과 `sass-embedded 1.69.3`을 사용한다. SassC 전환부터 해야 하는 저장소로 분류하지 않았다. 새 SCSS 및 의존성 조합의 실제 빌드는 이관 후 검증해야 한다.
- `html-proofer`는 `~> 4.4`에서 `~> 5.0`으로 바뀐다. 원본 스크립트의 `--disable-external` 등 새 호출 형식도 현재 workflow와 맞춰야 한다.
- 사용자 추가 Gem과 plugins를 보존해야 한다. 특히 `jemoji`, `jekyll-feed`, 로컬 serve용 `webrick`을 원본 Gemfile 전체 선택으로 잃지 않도록 한다.
- 원본 gemspec은 `jekyll-redirect-from` 의존성을 제거했다. 현재 글에서 `redirect_from`/`redirect_to` front matter는 발견되지 않았으므로 기존 글만을 위해 무조건 다시 추가해야 한다고 판단하지 않았다.
- 새 package는 Bootstrap 5.3 계열, Rollup 4, 새 lint 도구 등을 사용하며 jQuery 중심 테마 코드가 Vanilla JS로 바뀐다. 새 패키지 설치와 해당 Node 버전 호환 확인이 필요하다. 이 분석에서는 새 npm 의존성을 설치하거나 실행하지 않았다.
- `assets/lib` 서브모듈 포인터도 변경된다. 현재 `self_host.enabled`는 비어 있어 CDN 경로를 사용한다. 자체 호스팅을 켤 경우 새 라이브러리 파일과 서브모듈 checkout을 별도로 확인해야 한다.

### 6.3 GitHub Actions

배포 workflow는 내 쪽에서 `.github/workflows/pages-deploy.yml.hook` → `.github/workflows/pages-deploy.yml`, 원본에서 같은 파일 → `.github/workflows/starter/pages-deploy.yml`로 각각 이동해 이름 변경 충돌이 난다. 기존 main 배포 workflow를 기준으로 필요한 최신 변경을 이관해야 한다.

원본의 CI·CodeQL·commitlint·stale 등은 사용자가 삭제한 파일과 충돌하고, 새로운 lint/publish/PR 관련 파일 일부는 충돌 없이 추가된다. 이들은 테마 개발용이므로 블로그 배포를 위해 모두 복구할 필요는 없다. 유지할 workflow를 골라야 불필요한 작업과 실패 알림을 피할 수 있다.

Pages의 저장소 설정은 Git 파일 병합과 별개다. 앞서 변경한 GitHub Actions 방식의 Pages 설정이 원본 동기화만으로 branch 방식으로 되돌아가는 것은 아니다. “368 behind”라는 숫자 자체도 새 글을 배포하지 못하게 막는 조건이 아니다.

### 6.4 새 사이트 초기화 스크립트는 업그레이드 절차로 실행하지 않음

원본의 `tools/init.sh`는 새 사용자를 위한 초기화다. 코드에 `git reset --hard`, `git clean -fd`, `_posts/*` 삭제, timezone/CDN/avatar 초기화가 포함되어 있다. 기존 글과 설정을 유지하는 테마 업데이트에 그대로 적용하면 안 된다. 일반 병합이 이 스크립트를 자동 실행하지는 않으며, 이번 분석에서도 실행하지 않았다.

근거: [초기화 스크립트](https://github.com/cotes2020/jekyll-theme-chirpy/blob/ae1057257b3927d6474deb549bdfafa71244e080/tools/init.sh).

## 7. 실제 동기화 시 권장 순서와 완료 기준

1. 현재 커밋·미추적 자료를 보존한 별도 브랜치/복사본에서 작업한다. 비교 대상 upstream 커밋을 고정한다. 현재 로컬의 `upstream` 이름은 원본 테마가 아니라 내 저장소 URL을 가리키므로, 이름만 믿고 동기화하지 않는다.
2. 개인 설정·메뉴·아이콘·수식·댓글을 유지하면서 최신 레이아웃을 이관한다. 삭제했던 데모 글과 원본 개발 workflow를 무조건 복구하지 않는다.
3. Gem과 npm 의존성, 생성 파일 관리 방식, Actions의 production 빌드 절차를 맞춘다.
4. 글 44편, 이미지 389개와 보조 파일들의 내용 보존을 재검사한다. 기존 글의 URL·categories/tags·검색·feed·댓글·About·SNS 링크를 확인한다.
5. `bundle exec jekyll serve`로 홈과 대표 글을 브라우저에서 확인한다. UMM처럼 긴 글에서 figure/table, `$`·`$$`, 기호 설명, quote, H1/H2/H3 목차, HTML 토글, 모바일 폭과 이미지 확대를 확인한다.
6. `JEKYLL_ENV=production bundle exec jekyll build` 및 production 결과의 브라우저 확인을 별도로 수행한다. 생성 CSS/JS, PWA, 수식의 새로고침, 404·콘솔 오류를 확인한다.
7. 사용자가 변화한 화면과 보존 결과를 확인한 뒤 해당 변경의 푸시를 승인하면 배포한다.

이는 향후 동기화 작업의 절차이며 이번 분석에서 수행 완료한 것으로 기록하지 않는다. 이번 작업은 비교·병합 모의 계산·파일 보존 검증과 문서 작성까지다.

## 부록: 충돌 관련 경로 전체 34개

이름 변경 충돌 1건의 이전 경로와 양쪽 새 경로가 함께 포함되어 32건보다 경로 수가 2개 많다.

```text
.github/DISCUSSION_TEMPLATE/general.yml
.github/ISSUE_TEMPLATE/bug_report.yml
.github/dependabot.yml
.github/workflows/cd.yml
.github/workflows/ci.yml
.github/workflows/codeql.yml
.github/workflows/commitlint.yml
.github/workflows/pages-deploy.yml
.github/workflows/pages-deploy.yml.hook
.github/workflows/stale.yml
.github/workflows/starter/pages-deploy.yml
.gitignore
Gemfile
README.md
_config.yml
_includes/sidebar.html
_layouts/default.html
_layouts/post.html
_posts/2019-08-08-text-and-typography.md
_posts/2019-08-08-write-a-new-post.md
_posts/2019-08-09-getting-started.md
_posts/2019-08-11-customize-the-favicon.md
assets/feed.xml
assets/img/favicons/android-chrome-192x192.png
assets/img/favicons/android-chrome-512x512.png
assets/img/favicons/apple-touch-icon.png
assets/img/favicons/browserconfig.xml
assets/img/favicons/favicon-16x16.png
assets/img/favicons/favicon-32x32.png
assets/img/favicons/favicon.ico
assets/img/favicons/mstile-150x150.png
assets/img/favicons/site.webmanifest
docs/CHANGELOG.md
docs/SECURITY.md
```
