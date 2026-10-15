# 블로그 디자인 버전과 복원 방법

디자인 변경은 글 본문·논문 이미지·SEO 수정과 분리된 커밋으로 보관한다. 태그는 해당 시점의 디자인을 가리키며, 이미 발행한 태그를 다른 커밋으로 이동하지 않는다.

## 저장된 버전

| 버전 | Git 태그 | 커밋 | 설명 |
| --- | --- | --- | --- |
| 변경 전 | `design/before-2026-10-13` | `a2d9cac` | Chirpy 7.6.0의 기존 디자인. UniCompress·UMM Transferability 글과 최신 SEO 수정이 포함된 기준점이다. |
| 사용자 지정 디자인 | `design/2026-10-13` | `8c2986f` | 아래 디자인 변경을 하나의 독립적인 커밋으로 저장한 버전이다. |
| 카테고리 폭·날짜 정렬 | `design/2026-10-13.1` | `e91025a` | 트리를 처음부터 본문 최대 폭으로 표시하고 날짜를 오른쪽 끝에 정렬한다. 스크롤바 공간을 미리 확보하여 폴더를 펼쳐도 가로 위치가 유지된다. |
| 2026-10-14 사이드바 수정 전 | `design/before-sidebar-2026-10-14` | `8948edd` | 기존 사용자 지정 디자인과 카테고리 폭·날짜 정렬 수정이 포함되어 있다. |
| 사이드바 선택형 디자인 | `design/2026-10-14` | `e910a1d` | 아래의 사이드바 변경 28개 파일과 빌드 산출물을 하나의 독립 커밋으로 저장한 버전이다. |
| 원형 배치 수정 전 | `design/before-orbit-2026-10-15` | `4d123c6` | 직전 배포 상태. 사진 위의 가로 배열과 8개 배경 선택 기능, 기존 테마·연락처 디자인을 포함한다. |
| 원형 배치·관성 회전 | `design/2026-10-15` | `c1b625d` | 사진 둘레의 원형 배치, 드래그·관성 회전, 각도 저장과 관련 빌드 산출물 13개 파일을 하나의 독립 커밋으로 저장했다. |

사용자 지정 디자인에 포함된 변경:

- 글 목록의 검은색 Magic Card 효과.
- 사이드바의 타이핑 문구, 인터랙티브 격자, PC 접기, System/Custom 테마 선택과 내부 해·달 스위치.
- GitHub·Instagram·메일·RSS의 유리 스타일 바. 컨트롤 영역에서는 격자 호버 강조를 숨긴다.
- 논문 리뷰의 읽기 진행률, 부연설명 토글, 작은 PDF·코드·프로젝트 버튼.
- 주제별 카테고리 트리. 학회 폴더를 건너뛰고 게시일 최신순으로 글과 날짜를 표시한다.
- 직접 사용한 UI 코드와 아이콘의 출처·라이선스 기록.

글의 Markdown, 카테고리 메타데이터, 논문 이미지와 최신 SEO 설정은 이 디자인 커밋에서 수정하지 않았다. 이 안내 문서는 디자인 커밋 다음의 별도 커밋으로 저장하므로 디자인을 되돌려도 유지된다.

2026-10-14 사이드바 버전에 포함된 변경:

- Home·Categories·Tags·Archives·About, 연락처 바, System/Custom을 같은 불투명한 배경과 입체적인 테두리·그림자로 통일.
- Custom 내부 해·달 스위치의 가로 폭을 80%로 축소하고, 아이콘 외곽선을 0.6px로 보강. 기존 기능과 강조색은 유지.
- 상단의 8개 배경 선택지: 기본, 인터랙티브 격자, 책장, 노트, 별빛, 바다, 수영장, 유령. 선택은 브라우저에 저장.
- 책장·별빛은 사이트의 라이트/다크 설정과 관계없이 원본 나이트 색상 유지.
- ABOUT 아래의 바다 지평선·해와 System/Custom 위의 DAWN/NIGHT 조절 바.
- ABOUT 아래, System/Custom 위의 인터랙티브 수영장. Three.js 0.180.0을 저장소 안에 포함하고 해당 배경에서만 로드.
- 같은 위치의 Ghost Jam 유령 GIF, 재생·일시정지 버튼과 진행 바. 음원은 버튼을 눌러야 시작하며 배경 전환·사이드바 닫기·탭 숨김 때 정지.
- 코드·의존성과 외부 미디어의 출처·라이선스 기록. 유령 GIF·음원은 원본의 외부 URL을 사용하므로 해당 호스트의 가용성에 의존.

검증: JavaScript·SCSS 검사, `npm run build`, production Jekyll 빌드, SEO 검사, 147개 HTML 파일의 내부 링크·이미지·스크립트 검사 통과. 홈·카테고리·리뷰 글의 사이드바와 PC·모바일 배치를 확인했다. 초기 GIF 자리에는 유효한 투명 이미지 주소를 두어 배포 검사를 통과하고, 유령 배경을 선택하면 실제 GIF를 불러온다.

## 2026-10-15 원형 배치·관성 회전

- 8개 배경 선택 버튼을 사진 중심에서 반지름 74px, 45도 간격으로 배치한다. 사진과 아이콘의 방향은 고정한다.
- 트랙이나 버튼을 잡아 드래그하면 배열이 회전한다. 놓을 때의 방향과 속도를 이어받아 시간에 따라 감속하며 멈춘다.
- 다시 잡기, 키보드 조작, 사이드바 닫기, 탭 숨김·페이지 이동 때 관성을 정지한다. 시스템의 동작 줄이기 설정에서는 관성을 생략한다.
- 드래그 후의 클릭이 배경을 바꾸지 않도록 구분하고, 일반 클릭·방향키 선택과 사진의 홈 링크는 유지한다.
- 회전 각도를 저장하고 페이지를 열 때 미리 반영하여 새로고침·페이지 이동 후에도 같은 배열을 유지한다.

검증: JavaScript·SCSS 검사, `npm run build`, production Jekyll 빌드, SEO 검사와 147개 HTML 파일의 내부 링크·이미지·스크립트 검사를 통과했다. 실제 브라우저에서 양방향 회전, 자연 감속·정지, 다시 잡기, 8개 배경 선택, 키보드 조작, 사진 고정과 새로고침 후 각도 유지를 확인했다. PC 및 375px·390px 모바일 크기와 라이트·다크 모드에서 배치와 스크롤을 확인했다. 모바일 관성 측정에서는 속도가 약 166도/초에서 28도/초로 감소한 후 정지했다. 별도 로직 검사에서는 30·60·120Hz 시간 간격에서 감속 결과가 일관됨을 확인했다.

### 원형 배치만 되돌려 가로 배열로 복원

다른 작업이 있다면 먼저 별도 커밋이나 stash로 보관한다. 다음 명령은 `c1b625d`만 되돌리므로 8개 배경, 기존 사이드바 버튼·테마 스타일과 이후의 글·SEO 변경을 보존한다.

```bash
git revert --no-commit design/2026-10-15
npm run build
git add -- _sass/vendors/_bootstrap.scss assets/js/dist/categories.min.js assets/js/dist/commons.min.js assets/js/dist/home.min.js assets/js/dist/misc.min.js assets/js/dist/page.min.js assets/js/dist/post.min.js
git diff --cached --check
JEKYLL_ENV=production bundle exec jekyll build
bundle exec jekyll serve
```

로컬 화면을 확인한 뒤 복원 커밋을 만들고, 사용자가 복원본의 푸시를 승인하면 원격에 반영한다.

```bash
git commit -m "revert(ui): restore the horizontal sidebar selector"
git push origin main
```

임시 Git 복사본에서 위 디자인 커밋을 `revert --no-commit`으로 되돌렸으며 충돌 없이 전체 파일 트리가 `design/before-orbit-2026-10-15`와 정확히 일치했다. 실제 작업 폴더는 복원하지 않았다. 이 안내 문서는 디자인 다음의 별도 커밋에 저장하므로 디자인을 되돌려도 유지된다.

## 2026-10-14 사이드바 변경도 복원

현재 버전에서 이 단계까지 돌아가려면 먼저 위 절의 원형 배치 복원을 완료하고, 이어서 아래 절차를 실행한다.

아래 명령은 `e910a1d`의 사이드바 변경만 되돌린다. 기존 카테고리 디자인과 이후의 글·SEO 변경을 보존하는 방식이다. `reset --hard`나 강제 푸시로 저장소 전체를 과거 시점으로 옮기지 않는다.

다른 작업이 있다면 먼저 별도 커밋이나 stash로 보관한다. 이후 다음 순서로 복원본을 준비한다.

```bash
git revert --no-commit design/2026-10-14
npm run build
git add -- _sass/vendors/_bootstrap.scss assets/js/dist/categories.min.js assets/js/dist/commons.min.js assets/js/dist/home.min.js assets/js/dist/misc.min.js assets/js/dist/page.min.js assets/js/dist/post.min.js
git diff --cached --check
JEKYLL_ENV=production bundle exec jekyll build
bundle exec jekyll serve
```

로컬 홈·카테고리·리뷰 글을 확인한 뒤 복원 커밋을 만든다. 사용자가 복원본의 푸시를 승인하면 원격에 반영한다.

```bash
git commit -m "revert(ui): restore the previous sidebar design"
git push origin main
```

2026-10-14 배포 시점에 이 버전의 단독 복원 검증도 완료했다. 당시 임시 Git 복사본에서 `git revert --no-commit design/2026-10-14`를 실행했을 때 충돌이 없었고, 복원된 전체 파일 트리가 `design/before-sidebar-2026-10-14`와 정확히 일치했다. 실제 작업 폴더는 복원하지 않았다. 후속 디자인 변경이 같은 파일을 수정했다면 복원 시 충돌을 검토한다.

## 이전 디자인으로 복원

사용자가 복원을 요청하면 디자인 커밋을 `revert`하여 새로운 복원 커밋을 만든다. 이후 추가된 글이나 SEO 커밋을 보존하면서 디자인 변경만 되돌릴 수 있다. 후속 디자인 수정이 같은 파일을 바꿨다면 충돌을 검토하여 필요한 부분을 복원한다.

저장소 루트에서 먼저 `git status --short`로 다른 작업을 확인하고, 필요한 변경을 별도 커밋이나 stash로 보관한 다음 진행한다.

아래 명령은 모든 디자인 수정이 적용된 최신 버전을 기준으로 한다. 이미 되돌린 커밋은 `revert` 목록에서 제외한다.

```bash
git revert --no-commit design/2026-10-15 design/2026-10-14 design/2026-10-13.1 design/2026-10-13
npm run build
git add -- _sass/vendors/_bootstrap.scss assets/js/dist/categories.min.js assets/js/dist/commons.min.js assets/js/dist/home.min.js assets/js/dist/misc.min.js assets/js/dist/page.min.js assets/js/dist/post.min.js
git diff --cached --check
JEKYLL_ENV=production bundle exec jekyll build
bundle exec jekyll serve
```

홈·카테고리·실제 리뷰 글을 로컬 브라우저에서 확인한 뒤 복원 커밋을 저장한다.

```bash
git commit -m "revert(ui): restore the previous blog design"
```

사용자가 복원본의 푸시를 승인하면 `git push origin main`을 실행하고 GitHub Pages 배포 결과를 확인한다.

카테고리 폭·날짜 정렬 수정만 이전 상태로 되돌리려면 위의 첫 `revert` 명령 대신 다음 명령을 사용한다. 나머지 사용자 지정 디자인은 유지된다. 이후 로컬 빌드·렌더링을 확인하고 복원 커밋을 저장하는 절차는 동일하다.

```bash
git revert --no-commit design/2026-10-13.1
```

## 버전 비교와 후속 수정

```bash
git diff design/before-2026-10-13 design/2026-10-13
git diff design/before-sidebar-2026-10-14 design/2026-10-14
git diff design/before-orbit-2026-10-15 design/2026-10-15
git log --oneline --decorate
```

앞으로의 디자인 수정도 글 작성과 분리하여 커밋한다. 배포 승인된 디자인 버전에는 새로운 날짜 또는 버전 번호의 태그를 만들고, 이 문서의 표에 기준 커밋과 변경 범위를 추가한다.
