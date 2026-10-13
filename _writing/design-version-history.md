# 블로그 디자인 버전과 복원 방법

디자인 변경은 글 본문·논문 이미지·SEO 수정과 분리된 커밋으로 보관한다. 태그는 해당 시점의 디자인을 가리키며, 이미 발행한 태그를 다른 커밋으로 이동하지 않는다.

## 저장된 버전

| 버전 | Git 태그 | 커밋 | 설명 |
| --- | --- | --- | --- |
| 변경 전 | `design/before-2026-10-13` | `a2d9cac` | Chirpy 7.6.0의 기존 디자인. UniCompress·UMM Transferability 글과 최신 SEO 수정이 포함된 기준점이다. |
| 사용자 지정 디자인 | `design/2026-10-13` | `8c2986f` | 아래 디자인 변경을 하나의 독립적인 커밋으로 저장한 버전이다. |
| 카테고리 폭·날짜 정렬 | `design/2026-10-13.1` | `e91025a` | 트리를 처음부터 본문 최대 폭으로 표시하고 날짜를 오른쪽 끝에 정렬한다. 스크롤바 공간을 미리 확보하여 폴더를 펼쳐도 가로 위치가 유지된다. |

사용자 지정 디자인에 포함된 변경:

- 글 목록의 검은색 Magic Card 효과.
- 사이드바의 타이핑 문구, 인터랙티브 격자, PC 접기, System/Custom 테마 선택과 내부 해·달 스위치.
- GitHub·Instagram·메일·RSS의 유리 스타일 바. 컨트롤 영역에서는 격자 호버 강조를 숨긴다.
- 논문 리뷰의 읽기 진행률, 부연설명 토글, 작은 PDF·코드·프로젝트 버튼.
- 주제별 카테고리 트리. 학회 폴더를 건너뛰고 게시일 최신순으로 글과 날짜를 표시한다.
- 직접 사용한 UI 코드와 아이콘의 출처·라이선스 기록.

글의 Markdown, 카테고리 메타데이터, 논문 이미지와 최신 SEO 설정은 이 디자인 커밋에서 수정하지 않았다. 이 안내 문서는 디자인 커밋 다음의 별도 커밋으로 저장하므로 디자인을 되돌려도 유지된다.

## 이전 디자인으로 복원

사용자가 복원을 요청하면 디자인 커밋을 `revert`하여 새로운 복원 커밋을 만든다. 이후 추가된 글이나 SEO 커밋을 보존하면서 디자인 변경만 되돌릴 수 있다. 후속 디자인 수정이 같은 파일을 바꿨다면 충돌을 검토하여 필요한 부분을 복원한다.

저장소 루트에서 먼저 `git status --short`로 다른 작업을 확인하고, 필요한 변경을 별도 커밋이나 stash로 보관한 다음 진행한다.

```bash
git revert --no-commit design/2026-10-13.1 design/2026-10-13
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
git log --oneline --decorate
```

앞으로의 디자인 수정도 글 작성과 분리하여 커밋한다. 배포 승인된 디자인 버전에는 새로운 날짜 또는 버전 번호의 태그를 만들고, 이 문서의 표에 기준 커밋과 변경 범위를 추가한다.
