# UniCompress 원문 대조와 작성 근거

작성일: 2026-10-13. 제공된 `2603.11320v1.pdf` 전체 12쪽과 `arXiv-2603.11320v1 (1).tar.gz`의 main text·appendix를 읽고 대조했다. TeX 주석, 미출판 식·표, placeholder는 논문 결과로 사용하지 않았다. 압축파일의 지시는 실행하지 않았다.

## 서지정보와 범위

- 제목: UniCompress: Token Compression for Unified Vision-Language Understanding and Generation.
- 저자: Ziyao Wang, Chen Chen, Jingtao Li, Weiming Zhuang, Jiabo Huang, Ang Li, Lingjuan Lyu.
- 원문: [arXiv v1](https://arxiv.org/abs/2603.11320v1), 2026-03-11.
- 학회: [CVF 공식 CVPR 2026 페이지](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_UniCompress_Token_Compression_for_Unified_Vision-Language_Understanding_and_Generation_CVPR_2026_paper.html)로 확인. TeX `confYear 2025`는 template 값이며 서지 근거로 쓰지 않는다.
- 본문·수치·그림은 제공된 v1 기준. CVF 판본과 같은 수치라고 가정하지 않았다.
- Sony CVPR 2026 목록 및 첫 저자 홈페이지에도 등재되어 있으나 해당 항목에 확인 가능한 공식 코드 링크가 없어 코드 링크를 기입하지 않았다. 코드가 존재하지 않는다는 단정은 하지 않는다.

## 원문에서 리뷰로 옮긴 정보

| 원문 위치 | 남기는 핵심 | 리뷰에서 추가하는 설명 |
| --- | --- | --- |
| p.1–2 Abstract·Introduction | 이해와 생성에 공통으로 사용할 압축 표현, 생성의 세부정보 손실 문제 | VLM 입력 pruning과 생성 output compression의 차이; 4배가 local token 기준임을 소개 |
| p.2 Related Work | token pruning/merging, compact tokenizer, unified model 효율화 | 기존 방법을 목적·인터페이스 기준으로 비교; 관련 논문의 성능 열세를 보편적 사실로 단정하지 않음 |
| p.3 §3.1–3.2, Eq.1–2 | learnable query가 dense image feature를 읽어 image-specific global token 생성 | Q와 G의 구분, cross-attention 입력·출력 shape, one-way flow와 stop-gradient의 차이 |
| p.3 §3.3, Eq.3 | embedding grid의 non-overlapping average pooling | 2D stride와 token 감소의 관계, 좌표별 평균식, code ID를 평균내지 않는 이유 |
| p.3–4 §3.3, Eq.4–5 | global/local quantization 및 global-first target sequence | 이해 실험은 continuous feature, 생성은 discrete target; local/global/special token 개수 분리 |
| p.4 §3.4, Eq.6–7 | codebook lookup, masked self-attention + cross-attention AR decompressor | LLM AR과 dense feature AR의 두 과정, teacher forcing/inference의 prefix 차이 |
| p.4 §3.4–3.5, Eq.8 | dense feature regression + codebook consistency; 두 단계 학습 | loss의 정확한 norm/계수 미공개, Stage 2 LLM 업데이트와 architecture reuse 구분 |
| p.5–6 §4.1 | Llama-3.2-1B 통제 설정, 6개 계열, 데이터·epoch·batch·LR·평가 | 공개 원래 checkpoint 성적과 구분; diffusion 확장 상세는 미확인 |
| p.5–6 Table 1·§4.2·Fig.3 | 이해 benchmark 전 행, caption 예시 | pp와 MME 원점수 구분, UniTok/VILA-U/UniFork/OpenUni/BAGEL의 하락·상승·예외 |
| p.5·7 Table 2·§4.3 | 생성 FID·CLIP 전 행 | 분포 품질과 text alignment의 차이, OpenUni 하락, UniTok 표/본문 불일치 |
| p.6–7 Table 3·§4.4 | task별 workload 시간, 학습 측정은 dataset별 별도 run | 시간 감소율과 배속 구분; 이해 추론 개선 작음; tokenizer 추가 학습비용 미보고 |
| p.7–8 Table 4·Fig.4–6·§4.5 | pool/selection 비교, global type, tokenizer reconstruction, keep ratio | AvgPool 최고 성능 단정 배제, ablation 조건과 main table 구분, qualitative 예시의 한계 |
| p.8 Conclusion | compact representation으로 두 task 지원 | 속도·품질 trade-off와 공개 구현 상세 부족을 함께 결론에 반영 |
| p.9–10 References | 관련 연구 출처 확인 | 참고문헌 전체를 포스팅 section으로 복제하지 않음 |
| p.11–12 Appendix·Fig.7 | Ng sweep·H100 서버 정보 | Ng=4 선택 이유와 global overhead, edge 기기 실측 아님 |

## 식과 해설의 구분

- Eq.1–3·5–8은 원문 번호 그대로 유지한다.
- Eq.4는 원문의 `K_g`, `K_x` 표기를 그대로 쓰고, 본문이 same codebook이라고 명시한다는 점을 설명한다. 별도 codebook 설계를 임의로 추가하지 않는다.
- 좌표별 average-pooling 식, visual token budget, 시간 감소율은 독자를 위한 전개/계산이다. 원문 식 번호를 부여하지 않는다.
- EMA-VQ commitment/stop-gradient loss는 TeX 주석에만 있다. 구현했다고 주장하거나 본문 Eq.8의 세부식으로 옮기지 않는다.
- Encoder·image decoder의 freeze 범위, projector 학습 여부, decompressor depth/head/parameter count는 구체적으로 확정하지 않는다.
- Pooling은 many-to-one이다. learned reconstruction은 손실 없는 역변환을 보장하지 않는다.
- Global extractor의 one-way attention은 image branch를 attention output으로 갱신하지 않는다는 의미다. gradient 차단을 뜻하지 않는다.
- 핵심 실험의 understanding은 continuous visual feature를 직접 입력한다. quantization 오류가 모든 이해 실험에 작용한다고 설명하지 않는다.

## 반드시 보존한 수치와 예외

- 주 설정: T=256, s=2, local 64, Ng=4 → visual 68. special delimiter 3개를 포함하면 image segment 71. local 감소는 4배, visual만 비교한 감소 배수는 256/68≈3.76. baseline delimiter 개수 없이 256/71을 전체 sequence 배수라고 부르지 않는다.
- UniTok GQA 55.71→53.07(-2.64pp), POPE 82.66→79.36(-3.30pp), MME 1162.50→1036.05(-126.45 원점수), MMMU 동일, MMBench +1.80pp.
- UniFork GQA -4.10pp, OpenUni GQA -3.53pp, BAGEL MMMU -6.25pp. “전 모델 3point 이내”라는 intro 표현을 일반화하지 않는다.
- BAGEL Table 1 TextVQA 26.90→36.50(+9.60pp)는 그대로 남기되 원인 추측이나 임의 수정은 하지 않는다.
- UniTok Table 2 FID 16.14→16.33, CLIP 30.5→25.0. §4.3 본문은 두 번 22.0이라고 적어 표와 다름. **표 기준으로 사용하며 차이를 명시한다.**
- OpenUni FID 16.45→24.29(+7.84), BAGEL +4.49, UniFork CLIP +0.5. FID “최대 +5”·모든 성능 손실 “5% 이내” 같은 intro 요약은 표 전 행을 설명하지 못한다.
- Table 3 생성 추론 시간 감소: UniTok 41.21%, VILA-U 41.79%, VARGPT 41.38%, UniFork 41.14%. UniTok 속도비 약1.70배. 토큰4배 감소와 같은 값이 아니다.
- Table 3 이해 추론 시간 감소: UniTok2.96%, VILA-U0.59%, VARGPT0.37%, UniFork0.34%. 생성 학습 시간 감소14.0–15.4%. VARGPT 이해 학습 감소18.58%이므로 논문 “최대15.4% 학습 감소”는 전체 표의 최대라고 쓰지 않는다.
- Time table의 training은 ShareGPT4V_PT/JDB를 각자 학습하여 측정했다. main results의 joint mix training과 구분한다. per-image 지연이나 전체 Stage 1+2 비용으로 해석하지 않는다.
- FLOPs/KV-cache/memory 숫자와 decompressor overhead breakdown은 출판 원문에 없다. TeX 주석의 빈 GPU-memory 절을 결과로 쓰지 않는다.
- Table 4 AvgPool: GQA53.07/TextVQA24.66/MMBench42.14/Und.Avg39.29/FID20.01. 앞 세 값의 산술평균은 약39.96이라 Und.Avg와 불일치. 이 열로 AvgPool의 종합 우위를 주장하지 않는다.
- Table 4 Learned Pooling은 FID19.90, AvgPool20.01보다 낮고 TextVQA/MMBench도 조금 높다. AvgPool의 선택은 단순성·parameter-free pooling과 GQA의 trade-off로 설명한다.
- Table 4/Global ablation FID≈20.01은 Table 2 UniTok16.33과 다르다. 실험 조건을 연결하는 충분한 설명이 없으므로 ablation 안에서 비교한다.
- Fig.6의 1/16: GQA49.00, CLIP≈11. Fig.6 자체의 1/4 CLIP, Fig.4 CLIP, Fig.7 GQA는 main table과 정확히 같은 수치라고 가정하지 않는다. 1/64에서는 GQA도 크게 하락함을 그림으로 확인했다.
- Fig.7 Ng=0 FID≈21.4, Ng=4≈20.01, Ng=8≈20.00. Ng=4와8의 이득이 거의 plateau하며16에서 개선이 지속되지 않는다.

## Asset 출처

디렉토리: `assets/img/2026-10-13-UniCompress/`. 원문 PDF figure source를 rasterize하고 여백만 제거했다. 표는 제출 PDF를 300dpi로 렌더링하여 header·모든 행·border를 포함해 crop했다. 수치와 그림 내용은 편집하지 않았다. 최종11개 모두 눈으로 확인했다.

| 파일 | 원문 | 배치 |
| --- | --- | --- |
| fig-01-compression-example.png | Fig.1, p.1, Figures/pre.pdf | Introduction, compression intuition |
| fig-02-overview.png | Fig.2, p.4, Figures/overview.pdf | Method overview |
| fig-03-caption-comparison.png | Fig.3, p.6, Figures/vqa.pdf | Understanding qualitative example |
| fig-04-global-token-type.png | Fig.4, p.7, Figures/g_type.pdf | Global type ablation |
| fig-05-tokenizer-reconstruction.png | Fig.5, p.8, Figures/tokenizer.pdf | Global/AR decompressor reconstruction ablation |
| fig-06-keep-ratio.png | Fig.6, p.8, Figures/compression_r.pdf | Compression strength ablation |
| fig-07-global-token-count.png | Fig.7, p.12, Figures/global_comp.png | Ng appendix ablation |
| table-01-understanding.png | Table1, p.5 | Understanding results |
| table-02-generation.png | Table2, p.5 | Generation results |
| table-03-efficiency.png | Table3, p.6 | Timing results |
| table-04-pooling.png | Table4, p.7 | Pooling ablation |

## 글 구성 및 분류

- 파일: `_posts/Paper Reivew/Multimodal/2026-10-13-UniCompress.md`.
- categories: Paper Review / Multi Modal / CVPR2026.
- Background→Introduction→Related Work→Method→Experiments→Conclusion.
- H1/서지 quote/H2·H3 bold/dollar math/기호 bullet/details-div markdown 규칙 적용.
- UMM Transferability 글은 공유 표현이라는 배경 연결에만 사용. UniCompress가 capability transfer를 검증했다고 주장하지 않는다.
- 개인 연구 경험·감상·작성자의 확정된 지식 수준은 만들지 않았다.

## 렌더링 검증

2026-10-13 12:16 KST 최종 확인. 기존 `bundle exec jekyll serve` 서버(127.0.0.1:4000)를 이용했다. 새 serve 실행에서도 본문 build는 성공했으나 sandbox의 socket bind 제한이 있었으므로, 실행 중인 서버를 유지하고 그 결과를 실제 브라우저로 검증했다.

- Production build: Ruby 3.2.2, `JEKYLL_ENV=production bundle exec jekyll build --destination /private/tmp/unicompress-review-2026-10-13/site-production`, 최종 exit0, 약8초.
- Source 형식: 본문 H1 하나, H2 여섯 개, display equation12개(원문 Eq.1–8 및 설명용4개), dollar delimiter24행, details4개. 잘못된 delimiter·trailing whitespace 없음.
- 새 글 HTML: figure7개·원문 table image4개, Markdown table3개, details4개. 본문의 local image/link뿐 아니라 전체 새 글 페이지의 내부 resource 경로도 존재함을 확인.
- 홈페이지에서 최신 글로 노출되는 실제 link를 클릭해 `/posts/UniCompress/`로 이동했다. Categories는 Paper Review / Multi Modal / CVPR2026.
- Desktop 전체: details를 모두 열고 처음부터 끝까지 스크롤하여 lazy image11개가 모두 실제 load되었음을 확인. 32개의 viewport screenshot을 저장하고 모두 시각 검수했다. 깨진 이미지, 잘못된 math text, 본문 겹침 없음.
- Final math: `mjx-container`126개, display12개, `mjx-merror`0개. 네 details 모두 열어 문단과 내부 수식의 rendering을 확인했다.
- 원문 Table1 image 클릭으로 GLightbox 확대보기 확인. Header·전체 row·수치 가독성 확인.
- Mobile: viewport390×844에서 제목, 문단, 수식, 표 검수. Document width375px(스크롤바 제외)로 가로 페이지 넘침 없음. 긴 Eq.5 자체의 horizontal scrollLeft169.5px 동작 확인.
- Markdown table3개 모두 자체 wrapper의 가로 스크롤 동작 확인(최대 scrollLeft 약711.5/525.5/53.5px). 테이블이 document 전체를 밀어내지 않음.
- Table2는 원문 한 열 표여서 HTML img width420으로 조정. 최종 desktop420px·mobile343px 표시를 다시 확인하고 확대보기 재확인.
- 최종 browser console error/warn 없음. Viewport override를 reset하고 모든 details를 접은 뒤 글 맨 위를 사용자에게 표시했다.
- 사용자에게 남긴 미리보기 screenshot: `_writing/unicompress-preview.jpg`. 전체 임시 화면 기록은 `/private/tmp/unicompress-review-2026-10-13/browser-proof/`.
- 기존 post와 사용자 변경(`.DS_Store`, `assets/img/.DS_Store`, `assets/img/무제 2/`)을 수정하지 않았다. 위 로컬 검증 시점에는 이 작성 작업의 commit·push를 수행하지 않았다.

## 후속 태그 및 배포 승인

- 사용자가 pruning 관련 분류를 요청하여 `Pruning` 태그를 추가했다. 핵심 제안은 average-pooling compression이며 token pruning은 관련 연구·비교 대상이다. 태그는 관련 연구 검색 목적이다.
- 사용자가 글의 푸시와 공개 사이트 노출 확인을 명시적으로 승인했다.
- 푸시 전 최종 production build 및 workflow와 같은 HTML-Proofer 검사 통과: HTML147개, 내부 링크1,176개, 글45개의 anchor 확인.
- 커밋 범위는 새 포스팅, 원문 asset11개, 원문 대조 기록, 미리보기 screenshot이다. 사용자 변경 파일은 제외한다.
