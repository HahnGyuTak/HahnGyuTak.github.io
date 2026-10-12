# UMM Transferability 원문 대조·asset 작업 기록

포스팅: `_posts/Paper Reivew/Multimodal/2026-10-12-UMM-Transferability.md`

본문 근거는 사용자가 제공한 `/Users/ghahn/Downloads/2607.04423v1.pdf`와 같은 버전의 `arXiv-2607.04423v1.tar.gz`다. PDF는 42페이지이며 본문은 pp. 1–15, references 이후 appendix는 PDF p. 24부터 시작한다. 아래 페이지 번호는 appendix에서 다시 시작하는 인쇄 페이지가 아닌 PDF 절대 페이지다.

공식 arXiv와 프로젝트 페이지로 ECCV 2026 발표 정보를 확인했다. 온라인 기본 arXiv 페이지에는 v2가 있지만 리뷰의 본문·수치·이미지를 그 버전으로 교체하지 않았다. 공식 GitHub 링크는 확인했으며 공개 저장소에는 검수 시점에 README가 표시되었다. 실행 코드를 검증했다고 쓰지 않았다.

- [리뷰 대상 v1](https://arxiv.org/abs/2607.04423v1)
- [발표 정보](https://arxiv.org/abs/2607.04423)
- [공식 프로젝트](https://cvlab-kaist.github.io/UMM_Transferability/)
- [공식 GitHub](https://github.com/cvlab-kaist/UMM_Transferability)
- 추가 배경의 1차 출처: [LoRA §4.1](https://arxiv.org/html/2106.09685v2), [Lumina-DiMOO](https://arxiv.org/html/2510.06308v1)

## 원문에서 본문으로 옮긴 정보

| 원문 위치 | 남긴 정보·배치 | 추가한 설명·주의점 |
| --- | --- | --- |
| Abstract, §1–2 | Background·Introduction: UMM, representation, task 간 전달, 학습 전략의 동기 | 텍스트 답변도 생성되지만 논문의 generation은 이미지 생성임을 구분; pretrained 단계의 supervision과 추가 학습을 구분 |
| §3, Appendix A | Method: 네 구조, 독립 und/gen run, LoRA 학습 범위 | 공유 transformer와 공유 이미지 표현을 분리; frozen head와 입력 미분을 구분; BLIP3-o의 branch 차이 |
| Appendix B.1 | Counting 학습·평가 데이터와 sample 수 | 동일 이미지가 동일한 모든 데이터 조건은 아님; count parsing fallback의 평가 범위 |
| Table 1 | 구조별 양방향 전달 | Janus-Pro 역방향 accuracy 상승과 MAD 악화; BLIP3-o의 단방향 개선 보존; architecture 인과 효과로 확대하지 않음 |
| §4.1, Table 2, Appendix C.3 | Counting 전달과 직접 생성 학습, FID/IS | FID reference는 baseline 생성; main table의 품질 수치는 COCO 조건; task prompt의 IS 상승 예외 |
| §4.2, Appendix B.2 | 대각선 데이터, 중심점의 부호, 관계 평가 | 검출 실패가 평가에서 제외됨; 250 MCQ understanding과 100×4 generation 구분 |
| §4.3, Appendix B.3 | OCR dataset·rendering·평가·수치 | H1·120자·영어 범위; und 4 template/gen 1 template 차이; direct CER/edit distance 악화와 transfer BLEU 악화 보존 |
| §5.1, Tables 5–6 | 역방향 전달과 transfer strength | 비교 비율의 식은 해설용으로 작성; 140% 때문에 upper bound는 보장된 상한이 아님; WER의 방향 예외 |
| §5.2, Fig. 8 | Joint training의 성능 교대 | gradient 충돌은 가설; post-training·데이터 특수성을 남김 |
| §5.3, Table 7 | 일반 이해 능력 | Counting MME-P 1534→1492, POPE 소폭 하락과 OCR MMBench 소폭 하락 보존 |
| Appendix C.1–2 | MMaDA·attention | 추가 모델의 일반성 근거와 제한; 두 모델 attention의 추출 layer·시점 차이 |
| Appendix C.4–5 | rank ablation·full FT 실패·능력 조합 | 3K step 결과와 main best checkpoint를 구분; composition 28.5%의 낮은 절대 수준 |
| Appendix E 및 평가 조건 | Conclusion의 Limitations | 원문 한계와 평가·인과 해석상 추가 판단을 근거와 함께 구분 |

논문에 새로운 번호 수식이 없는 분석 논문이므로 식 번호를 임의로 붙이지 않았다. LoRA·chain rule·Acc/MAD·좌표 조건·transfer strength의 식은 배경 또는 원문의 정의를 해설하기 위한 재표현이다. 핵심 주장을 긴 토글에 숨기지 않았으며, gradient 부연·모델별 설정·count parsing·task-specific IS 해석만 접었다.

## Asset 대응

저장 폴더: `assets/img/2026-10-12-UMM-Transferability/`

Figure는 제공된 TeX의 원본 vector PDF를 렌더링하고 흰 외부 여백을 정리했다. Table은 사용자 제공 PDF를 250 DPI로 렌더링하여 표 영역만 캡처했다. Table 5는 가독성을 위해 두 패널로 분리하며 OCR 패널에는 원문 행 제목 crop을 반복 배치했다. 숫자·색·열 제목을 다시 그리거나 바꾸지 않았다.

| 원문 | PDF 페이지 | Asset |
| --- | ---: | --- |
| Fig. 1 | 5 | `fig-01-architectures.png` |
| Table 1 | 5 | `table-01-architecture-transfer.png` |
| Fig. 3, Table 2 | 8 | `fig-03-counting-quality.png`, `table-02-counting.png` |
| Fig. 4 | 9 | `fig-04-spatial-dataset.png` |
| Fig. 5, Table 3 | 10 | `fig-05-spatial-quality.png`, `table-03-spatial.png` |
| Fig. 6–7, Table 4 | 11 | `fig-06-ocr-dataset.png`, `fig-07-text-quality.png`, `table-04-text-generation.png` |
| Tables 5–6 | 13 | `table-05-reverse-counting-spatial.png`, `table-05-reverse-ocr.png`, `table-06-transfer-strength.png` |
| Fig. 8 | 14 | `fig-08-joint-training.png` |
| Table 7 | 15 | `table-07-general-benchmarks.png` |
| Table 10 | 33 | `table-10-mmada.png` |
| Fig. 10 | 34 | `fig-10-attention.png` |
| Table 11 | 36 | `table-11-distribution-coco.png`, `table-11-distribution-task.png` |
| Table 12 | 37 | `table-12-lora-rank.png` |
| Table 13 | 38 | `table-13-composition.png` |

Table 3의 FID는 30.95, Appendix Table 11의 대응 값은 30.96으로 작은 차이가 있다. 본문에서는 각 표의 값을 유지하며 완전한 일치를 주장하지 않는다.

Fig. 2は構造比較とcounting定性結果の重複を避けて省略した。Appendixの追加定性例を全部貼り付けず、主張を支える代表図と追加分析を選択した。

## ローカル検証記録

- `bundle check`: dependencies satisfied。
- `bundle exec jekyll serve`: Jekyll 4.3.2で生成・127.0.0.1:4000のサーバー起動を確認。Sandbox内でのbind制限後、承認レビューを通った同じコマンドで起動した。
- ホームの新規リンクを実際にクリックし、`/posts/UMM-Transferability/`への遷移を確認。
- ソースのH1・front matter・math delimiter・HTMLの対応・assetパスを検査。
- 原文画像を目視し、Table 7・12の初期cropで切れた端を修正。Table 5は広すぎるため分割した。
- Desktopで全本文をscrollし、21画像すべてのnaturalWidthとロード成功を確認。トグル4つを開き、数式・表・画像・コード表記を確認。
- 390×844のviewportで再読み込みして検査。長い式をalignedで分け、documentの横幅超過を解消。比較Markdown表はサイト既存のスクロール容器内に収まり、トグル開閉も機能する。
- 数式エラー要素は0。MathJaxの旧CDNに関する既存warningはあるが、今回の式は描画されている。既存の`3d-editing` tag重複に関するbuild warningは今回のpostに由来しないため、既存記事を一括変更していない。
- GitHub commit/push는 수행하지 않았다. 사용자의 최종 컨펌과 명시적 push 승인을 기다린다.
