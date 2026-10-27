# Wall바둑 웹사이트

- 메인: https://wallbaduk.pages.dev/
- 기보 비교: https://wallbaduk.pages.dev/replays?game=16

HTML/CSS/JavaScript로 만든 정적 사이트다. 스크롤에 맞춰 실제 앱 에셋으로
기보를 재생하고, 마지막에 영역 계산과 승리 애니메이션을 보여준다.
공식 YouTube 클립은 재생 버튼을 누를 때 불러온다.

## 기보

메인은 선택한 후보 1(`games[16]`)의 22수 상태에서 마지막 8수를 진행한다.
최종 결과는 빨강 25칸, 파랑 24칸이다. 과거 온라인 기록이며 참가자가
사람인지 봇인지는 확인되지 않았다. 방송 영상의 기보와는 별개다.

비교 페이지는 `games[16]`, `games[26]`, `games[10]`, `games[42]`를 제공한다.
앞의 세 판은 과거 온라인 기록, 마지막 판은 봇 벤치마크다.
익명화된 상태와 행동만 포함한다. 원본 corpus 파일은 이 저장소에 포함하지 않는다.

비교 데이터를 다시 내보낼 때:

```sh
python3 scripts/export-candidates.py /path/to/real-game-corpus.json
```

## Cloudflare Pages 배포

새 빈 출력 디렉터리를 지정해 공개 파일만 묶는다.

```sh
python3 scripts/package-pages.py --output /tmp/wallbaduk-pages --base-url https://wallbaduk.pages.dev
npx wrangler pages deploy /tmp/wallbaduk-pages --project-name wallbaduk --branch main
```

Wrangler의 Cloudflare 로그인이 필요하다. GitHub 소스 반영과 Cloudflare 배포는
별도 작업이며, 현재는 직접 업로드 방식으로 배포한다.

패키징은 지정한 Pages 도메인의 루트용 `robots.txt`와 `sitemap.xml`도 생성한다.
사이트맵에는 대표 소개 페이지 하나만 포함하며, 스토어 도메인이나 기존 블로그의
URL은 넣지 않는다. 저장소 루트의 GitHub Pages용 검색 파일은 변경하지 않는다.
공개 후 두 파일이 HTML fallback이 아닌 텍스트와 XML로 열리는지 확인한다.
