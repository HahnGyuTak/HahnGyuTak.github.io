# 논문 원문에서 블로그 글로 옮기는 방식 분석

이 블로그는 논문의 내용을 짧게 전달하기보다, 작성자가 방법의 논리와 작동을 다시 이해할 수 있게 재구성한다. 가장 반복적인 변환은 배경을 앞에 보충하고, 모듈·수식·단계를 풀어 설명하며, 실험을 설계의 근거로 읽어주는 것이다. 향후 글에서는 이 설명 방식을 유지하고, 기존 글의 오기나 근거보다 강한 표현은 재현하지 않는다.

분석일은 2026년 10월 12일이다. 실제 작성에 적용할 규칙은 [작성 기준](paper-review-style.md)에 분리했다. **내용별로 어떤 Markdown/HTML 문법을 쓰는지, 실제 소스 예시와 과거 변형, 전체 39편의 본문 문법 집계는 [Markdown 작성 명세](paper-review-markdown-format.md)에 상세히 기록했다.**

## 분석 범위와 근거

`_posts/Paper Reivew`의 Markdown 포스팅은 총 39편이다. 전체 파일의 front matter, 제목 계층, 분량, 수식 delimiter, 이미지·코드·내부 링크 등의 형식을 조사했다. 원문을 먼저 읽고 포스팅 본문 전체를 대조한 대상은 **front matter 게시일 기준 최신 7편**이다. 나머지 32편을 모두 원문과 대조했다고 해석하면 안 된다.

7편은 논문의 초록만이 아니라 본문 전체, 수식·표·그림과 해당 글의 설명에 필요한 보충자료를 확인했다. 논문의 참고문헌 전체를 별도 논문으로 확장해 읽거나 모든 부록을 동일한 깊이로 분석한 것은 아니다. 그림은 포스팅의 로컬 이미지 참조 78개를 중복 포함해 확인했으며, 핵심 방법·평가 그림은 원문 페이지와도 대조했다. 이는 포스팅의 최종 웹 화면 렌더링 검증과는 구별된다.

| 글 | front matter 게시일 | 원문 대조 기준 | 원문에서 추가 확인한 범위 |
| --- | --- | --- | --- |
| Diffusion Bridge | 2026-02-19 | CVPR 2025 공식 공개본, 본문 pp. 1–8 | supplementary의 결합 실험·scale·추론 비용·데이터 크기 분석 |
| VLV AE | 2026-01-14 | arXiv 2507.07104 v1 본문을 읽고 v2와 변경 대조, 본문 pp. 1–9 | v2 Appendix A의 데이터·캡션 길이 통계; v1/v2 본문의 의미 차이는 없고 링크·인용 번호 등이 바뀜 |
| SD3 | 2025-12-15 | arXiv 2403.03206 v1, 본문 pp. 1–12 | Appendix B.1–B.4의 FM/CFM 유도, representation과 학습·평가 설정 |
| VIRAL | 2025-09-17 | arXiv 2509.07979 v1, 본문 pp. 1–11 | Appendix A와 B의 구현·token permutation·추가 시각화; 게시 이후 v2와 혼동하지 않음 |
| CoIBA | 2025-09-09 | CVPR 2025 공식 공개본, 본문 pp. 1–8 | supplement의 평가 항목·추가 결과 구성과 계산 비용 |
| VAR | 2025-09-05 | arXiv 2404.02905 v2, 본문 pp. 1–12 | Appendix A/B의 token dependency와 복잡도 |
| SigLIP | 2025-09-02 | arXiv 2303.15343 v4, 본문 pp. 1–10 | Algorithm 1, 초기값, 전체 결과·ablation; 부록 전체의 국가·언어별 숫자는 분석 범위 밖 |

날짜는 파일명보다 front matter를 우선했다. SD3는 파일명 `2025-12-01`과 게시일 `2025-12-15`, CoIBA는 파일명 `2025-09-05`와 게시일 `2025-09-09`가 다르다. 전체 목록과 나머지 글의 조사 수준은 [포스팅 목록](paper-review-inventory.md)에 있다.

## 전체 형식에서 관찰한 흐름

분야별 파일 수는 Computer Vision 18편, Multimodal 13편, 3D Vision 5편, NLP 2편, Audio 1편이다. 제목만으로도 생성 모델·멀티모달·3D를 반복적으로 읽는 경향이 보인다. 분야 이름은 파일이 놓인 폴더를 기준으로 센 것이며 연구 주제의 배타적인 분류는 아니다.

- 초기 글은 Attention, Encoder, Decoder처럼 구성요소를 따라 설명하거나 원문 장 번호를 유지한다.
- 2024년에는 GAN·diffusion의 수식 유도, 기존 글 내부 링크, Colab·Code Review가 구조상 자주 나타난다.
- 2025년 상반기는 Introduction → Method → Experiments의 흐름을 유지하면서 3D·멀티모달·이미지 편집으로 주제가 확장된다.
- 최근 표본 7편은 모두 Background를 Introduction보다 앞에 두고, Goal·Motivation·Contribution을 명시하며 마지막에 Conclusion을 둔다. 7편 중 6편에는 별도의 limitation 내용이 있다.
- 최신 7편의 원본 Markdown은 약 1.2만~3.2만 자다. 의미 있는 설명 문단이 짧게 분리되어 있으며 한 가지 해석을 여러 단계로 나누는 경향이 있다. 이 수치는 공백·메타데이터 등을 포함한 소스 길이이지 본문 단어 수가 아니다.
- 방법 이름을 최상위 장으로 삼거나, 배경·실험·한계를 이동하는 예외가 있으므로 고정 목차나 글자 수를 강제하는 스타일은 아니다.

구조 조사에서 추가로 DINO의 `쓰면서 생긴 Question`, ResCLIP의 `Code`, UniReal의 `개인적으로 느낀점` 장을 확인했다. 이 세 장의 편집·서술 방식은 참고했지만, 이번 분석에서 이 세 논문 전체를 원문 대조한 것은 아니다.

## Diffusion Bridge의 변환 방식

[블로그 글](../_posts/Paper%20Reivew/Multimodal/2026-02-19-DiffusionBridge.md) · [공식 원문](https://openaccess.thecvf.com/content/CVPR2025/papers/Lee_Diffusion_Bridge_Leveraging_Diffusion_Model_to_Reduce_the_Modality_Gap_CVPR_2025_paper.pdf) · [보충자료](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Lee_Diffusion_Bridge_Leveraging_CVPR_2025_supplemental.pdf)

초점은 **이미지와 텍스트가 같은 embedding 공간에 있어도 서로 교환 가능한 입력은 아니라는 문제**다. 이미지 생성용 diffusion과 다른, embedding 분포의 prior로서 diffusion을 읽는다. 설명은 최근 표본 중 비교적 간결하고 “이 절의 실험 목적”을 먼저 쓰는 방식이 특히 명확하다.

| 원문 위치 | 포스팅에서의 처리 | 무엇을 남기거나 더 설명했는가 |
| --- | --- | --- |
| Introduction와 modality gap 문제 | Background의 Modality Gap으로 먼저 이동 | 공유 공간과 분포 정렬을 구분하고 decoder 입력 분포가 달라지는 문제를 설명 |
| zero-shot captioning의 학습 전제 | 별도 Background 소절로 확장 | downstream 학습에서 image-caption pair를 쓰지 않는다는 뜻과 noise·memory 전략의 역할을 설명 |
| DDPM과 decoder augmentation | DDPM/DDIM을 짧게 선행 소개 | 두 sampling 방법이 각각 어디에서 쓰이는지 먼저 준비 |
| Abstract와 Introduction | 개념도와 Goal·Motivations·Contributions | prior, 중간 timestep 삽입, augmented training, memory bank 불필요라는 중심 아이디어를 묶음 |
| Related Work | Late/Early/Keywords Guidance 세 계열 보존 | 역사적 나열보다 개입하는 단계와 자기 방법의 위치를 설명; 생성 모델 관련 연구 나열은 압축 |
| §3.1 Eq. 1–5 | DDPM 정의와 기호 목록 | 원문의 기본 식을 대부분 유지하고 본 논문의 데이터가 픽셀이 아닌 text embedding임을 강조 |
| §3.2 Eq. 6 | 전처리 식과 이유를 인용문으로 해설 | normalization, centering, scaling을 나누고 alignment noise의 가우시안 근사와 score 직관을 연결 |
| §3.3 Eq. 7/8 | augmentation의 두 단계와 AR loss | 원복이 유일하지 않다는 이유를 decoder 강건성과 연결 |
| §4 구현 상세 일부 | Decoder 입력 어댑터로 Method에 이동 | projection, prefix projector, soft prompt의 역할과 차원 변환을 보충 |
| §3.4, Fig. 2 | Inference Process를 따로 설명 | 완전 noise가 아닌 이미지 embedding에서 중간 역확산을 시작하는 이유를 강조 |
| §4.1, Fig. 3, Table 1 | UMAP과 paired/unpaired similarity를 분리 해석 | 대표 값과 텍스트 간 비교를 남기고 projector가 unpaired 유사도도 높이는 현상에 기하적 설명 추가 |
| §4.2, Table 2/3 | In-domain과 cross-domain 분리 | 우수한 결과뿐 아니라 작은 데이터와 entity guidance의 장점도 남김 |
| supplement §1 | cross-domain 끝의 짧은 보충 인용문 | 결합 가능성을 실제 plug-and-play 근거와 연결 |
| §4.3, Table 4/5 | 구성요소와 timestep ablation 분리 | 추론 정렬의 효과와 augmentation의 추가 효과, timestep의 비단조 성능을 구분 |
| Conclusion | 핵심 경로를 한 문단으로 회수 | 마지막에는 반복적인 선전이나 새 전망을 크게 늘리지 않음 |

**그림 선택**: Fig. 1, Fig. 2의 세 단계 crop, Fig. 3과 Table 1–5를 모두 사용한다. 본문의 방법·평가 증거를 거의 그대로 보존하면서 긴 pipeline 그림을 단계별 문단 옆에 나눈다. supplement의 qualitative caption 예시는 사용하지 않는다.

**축약 또는 생략**: DDPM의 전체 학습 횟수·optimizer 설정·데이터별 A/M 등의 상당 부분, supplement의 scaling factor sweep, 실제 추가 GFLOPs·시간, 데이터 크기 실험은 본문에 자세히 옮기지 않는다. 효율성이 주요 주장인 후속 글에서는 추론 timestep M과 DDIM의 실제 평가 횟수, 비용을 함께 설명해야 한다.

**추가 해설의 경계**: projector를 `W`로 보고 metric을 `WᵀW`로 해석하는 부분은 글에서 추가한 단순화다. 실제 비선형 prefix projector의 현상을 이 식 하나로 증명한 것으로 읽으면 안 된다. 텍스트의 고밀도 영역으로 이동한다는 직관 역시 모든 이미지 의미 보존의 보장이 아니다.

**향후 적용**: 분포·정렬 논문에서는 공유 공간, 실제 분포, decoder가 학습한 입력을 구분하고, 정렬 자체와 decoder 강건화의 이득을 별도 실험으로 연결한다.

## VLV AE의 변환 방식

[블로그 글](../_posts/Paper%20Reivew/Multimodal/2026-01-14-VLVAE.md) · [원문 v2](https://arxiv.org/pdf/2507.07104v2)

초점은 **생성 모델의 지식을 재사용하여 언어에 가까운 representation을 효율적으로 학습하는 것**이다. embedding을 caption으로 읽는 경로뿐 아니라 다시 이미지로 복원하고, 공간·스타일을 합성하는 경로까지 다룬다.

| 원문 위치 | 포스팅에서의 처리 | 무엇을 남기거나 더 설명했는가 |
| --- | --- | --- |
| autoencoding과 diffusion representation 관련 개념 | Background의 VAE와 Representation Learning으로 앞당김 | 일반 배경을 추가하고 related work를 frozen teacher·distillation·denoising·vision-language 확장으로 분류 |
| Abstract/Introduction | 문제 → 재사용할 세 모델 → 병목 → 효율의 순서 | 생성 지식을 거꾸로 활용하는 동기와 연속 embedding의 장점을 반복 해설 |
| Introduction의 contributions | Goal·Motivation·Contributions로 재구성 | 비용 효율, 대규모 paired data 의존 감소, decoder의 역할을 강조; emergent properties의 기여는 뒤의 실험에서 회수 |
| §3.1, Fig. 2 | Pipeline Overview | VLV encoder, diffusion decoder, caption decoder를 각각 역할·연결로 설명하고 전체 두 단계 그림 삽입 |
| §3.2 Eq. 1 | Stage 1의 순서 목록과 loss 해설 | 학습되는 encoder가 고정된 generator를 통해 설명 가능한 조건을 만들어야 한다는 직관 추가 |
| §3.2의 세부 token flow | 그림과 서술 중심으로 압축 | 원문의 모든 shape·dummy prompt·query와 Transformer encoder/decoder 연산을 수식으로 옮기지는 않음 |
| §3.3 Eq. 2 | Stage 2의 `z → T(z) → ψ(c) → LLM` | prepend 전후 시퀀스, 실제 단어 위치만 loss에 포함, 장문 caption을 생성할 수 있는 이유를 해설 |
| §4.1 및 Appendix A | Dataset/Training Details | 두 단계의 데이터 종류·규모, filtering, 자동 caption 출처와 저비용 강조; 상세 optimizer보다 구조와 비용을 선택 |
| §4.2.1, Fig. 3, Table 1 | 목적·방법·데이터·결과·정성 평가로 분해 | caption을 통해 T2I reconstruction하는 평가 경로를 설명하고 embedding reconstruction과 함께 보여줌 |
| §4.2.2, Table 2 | Captioner Arena | 누가 무엇을 평가하는지, coverage·hallucination·spatial consistency를 풀어 설명 |
| §4.2.3, Table 3 | Text-only Question-Answering | 이미지가 아니라 caption만 제공되는 조건과 zero/few-shot 차이를 명시 |
| §4.3, Table 4 | query 수와 unfreezing을 한 소절로 묶음 | 병목 용량과 학습 모듈 수의 효과를 분리 설명 |
| §4.3, Table 5 | Scalability Analysis | 데이터와 decoder 크기를 늘리는 두 축을 별도 목록으로 정리 |
| §4.4.1, Fig. 4/5, Table 6 | 3D Visual Awareness | representation의 공간적 보존을 정량·정성으로 따로 설명 |
| §4.4.2, Fig. 6 | Multi-image Semantics | concat 조작을 스타일 전이·착용·객체 합성이라는 구체 사례로 읽어줌 |
| Conclusion/Limitations | Conclusion과 확장된 Limitations | pretrained 의존·데이터 편향·범용성·모델 간 격차를 글의 추가 판단까지 포함해 정리 |

**그림 선택**: Fig. 2–6과 Table 1–6의 정보가 대부분 포함된다. 공간 인식은 Fig. 5와 Table 6를 한 이미지로 붙여 비교한다. 앞부분의 cost/throughput 그림, 부록의 data filtering·caption 길이 histogram과 긴 qualitative caption 예시는 본문에서 이미지로 보여주지 않는다.

**추가 설명**: prepend 시퀀스, 두 단계에서 필요한 데이터의 차이, frozen decoder가 encoder를 제한하는 의미가 원문보다 길다. Stage 1 설명은 reconstruction이라는 직관과 실제 denoising objective를 연결하려는 방식이다. 이 연결 자체는 향후에도 유지할 가치가 있다.

**정확성 주의**: 글의 “VAE 기반 VLV encoder가 `z_t`를 더 잘 생성하도록 학습”이라는 설명은 원문의 conditioning embedding 경로와 diffusion latent 경로를 혼동한다. Stage 1에서 학습되는 핵심은 condition `z`를 만드는 VLV encoder다. Stage 2 caption 생성 모델을 GPT-4 계열이라고 쓴 부분은 Dataset 소절의 Gemini 설명 및 원문과 불일치한다. 226.8은 caption의 **token** 평균이지 문자 평균이 아니다. 원문 40M/10B와 “10억 장의 0.4%”도 단위가 맞지 않는다.

**생략과 확대**: Florence-2 초기화, 77 query, 학습 step·precision 등의 설정은 상당 부분 일반화한다. 원문의 OCR 데이터 부족, 오래된 SD2.1 teacher, video 확장은 글의 한계와 완전히 같은 목록이 아니다. “VQA에 직접 응용되지 않는다”는 추가 판단은 앞서 실은 text-only VQA와 범위를 구분해야 한다. 비용 비교도 재사용한 pretrained 모델의 전체 학습 비용까지 포함한 것으로 확대하면 안 된다.

**향후 적용**: generator를 teacher로 사용하는 논문에서는 고정 모델을 통과하는 gradient, condition과 latent의 구별, 두 단계의 데이터·모듈 상태를 최우선 설명한다. 응용 가능성은 실제 검증한 task와 분리한다.

## SD3의 변환 방식

[블로그 글](../_posts/Paper%20Reivew/Multimodal/2025-12-01-SD3.md) · [원문](https://arxiv.org/pdf/2403.03206v1)

초점은 **flow/diffusion의 공통 수학과 multimodal Transformer 설계, 실제 대규모 학습의 안정성**이다. 이 글은 짧은 요약으로는 설명할 수 없는 곳에 장문 유도와 접힌 보충을 배분하는 대표 사례다.

| 원문 위치 | 포스팅에서의 처리 | 무엇을 남기거나 더 설명했는가 |
| --- | --- | --- |
| RF와 비교 formulation | RF·CFG·EDM Background | 원문에 별도 tutorial로 없는 CFG까지 비교 실험 이해를 위해 추가 |
| Abstract/Introduction | 대표 생성 결과와 Goal·Motivation·Contributions | sampling 효율, text 이해, multimodal weight, scaling 세 축을 유지 |
| §2 Eq. 1–8 | Simulation-Free Flow Training | CNF의 solver 비용, conditional path와 marginal path의 차이, 왜 CFM으로 학습할 수 있는지 확장 |
| §2 Eq. 4–6 | 조건부 벡터장의 좌표 변환과 Bayes 해설 | 원문의 짧은 식을 inverse map·시간 미분·혼합 분포의 posterior 평균으로 풀어 설명 |
| Appendix B.1 Eq. 30–36 | 접힌 FM → CFM 유도 | 제곱 전개, 파라미터와 무관한 상수, 기댓값의 교환과 marginalization을 단계별 설명 |
| §2 Eq. 9–12 | noise prediction으로의 재파라미터화 | 미분·대입·log-SNR를 통해 익숙한 diffusion MSE와 연결 |
| §3 | RF·EDM·Cosine·LDM-Linear 비교 | 같은 `a_t, b_t, w_t` 언어로 어떤 선택이 달라지는지 정리; 세부 식은 선택적으로 유지 |
| §3.1 Eq. 18–22, Fig. 11 | Tailored SNR Samplers | 어려운 중간 timestep, 분포 파라미터 효과, sigmoid sampling 의사코드, mode와 CosMap의 차이 |
| §4, Fig. 2, Appendix B.2 | MM-DiT의 Encoding과 block | image와 text crop을 나누고 pooled/global와 contextual/token-level 입력을 구분 |
| §4의 separate weights와 joint attention | 접힌 Cross Attention 비교 | Q/K/V 식과 시퀀스 관점으로 정보 교환 차이를 해설 |
| §5.1, Table 1/2, Fig. 3 | 61 formulation 비교와 24 조건의 ranking | 지표·EMA·데이터·sampling 조건을 나누고 일부 설정의 trade-off와 실패도 남김 |
| §5.2.1, Table 3, Fig. 10 | Autoencoder Latent dimensions | reconstruction 상한과 생성 모델 용량의 trade-off를 연결 |
| §5.2.2, Table 4 | Caption | 합성 caption 혼합을 데이터의 정보량·개념 망각 문제와 연결 |
| §5.2.3, Fig. 4 | Backbone 비교 | DiT/CrossDiT/UViT와 two/three-stream 비교; 더 많은 weight의 비용을 남김 |
| §5.3.1 | preencoding 위주로 압축 | frozen encoder 출력이 일정하므로 사전 계산 가능하다는 이유를 설명 |
| §5.3.2, Fig. 5 | QK-Normalization | logit scale, softmax 집중, entropy, RMSNorm과 bf16 안정화를 연결 |
| §5.3.2 | varying aspect ratio position encoding | bucket·큰 grid·center crop의 좌표 과정을 서술로 복원 |
| §5.3.2 Eq. 23–25, Fig. 6 | resolution-dependent timestep shifting | 상수 이미지 가정에서 평균·분산·표준오차를 계산하고 해상도 shift를 직접 유도 |
| §5.3.3, Fig. 7/8, Table 5 | 최종 결과 | validation loss와 사람·자동 지표의 관계를 강조 |
| Flexible Text Encoders, Fig. 9 | 별도 소절 | dropout이 encoder subset 사용을 가능하게 하는 이유와 T5가 필요한 prompt를 설명 |
| Conclusion | Conclusion과 추가 Limitation | 연산·메모리·1-step 미달성·검증 범위를 추가 판단으로 정리 |

**그림 선택**: Fig. 2를 전체 그림과 세부 crop으로 반복 사용한다. RF sampler 분포, ranking, latent·caption 비교, 안정성·shift·scaling·T5 차이까지 19개 이미지 참조로 따라간다. Table 6의 모델 크기별 sampling 효율 숫자와 Appendix C의 DPO 유도는 별도로 해설하지 않는다.

**수식 확장의 핵심**: 논문 본문의 식뿐 아니라 appendix 증명을 본문 근처로 가져오고, “이 식이 왜 필요한가”를 다음 장 선택과 연결한다. 특히 해상도 shift는 추정량부터 표준편차까지 원문보다 자세히 유도하며 비현실적인 상수 이미지 가정도 명시한다.

**축약**: 대규모 데이터의 filtering·deduplication, DPO, video 모델의 구체적인 temporal attention 구조, 전체 학습 hyperparameter 등은 압축되거나 생략된다. 글은 보안·데이터 파이프라인보다 loss·architecture·stability에 설명을 집중한다.

**정확성 주의**: 서지 인용문 제목은 VIRAL 제목이 남은 복사 오류다. 원문의 8B는 **80억**이며 여러 곳의 “8억”은 잘못됐다. FM Eq. 7은 marginal target `u_t(z)`여야 하는데 글에는 conditional target이 섞인다. RF weighting의 `t/(t-1)`는 원문 `t/(1-t)`와 부호가 다르다. logit-normal sampling은 normal 표본에 **sigmoid**를 적용하는 것이며 logit을 적용하는 것으로 쓰면 안 된다. MM-DiT 비교를 단일 `ZW_Q`로만 쓰면 별도 modality weight라는 핵심 차이가 가려진다. 일반 DDPM도 닫힌 형태의 noising 표본을 이용할 수 있으므로 “기존 diffusion 학습은 모두 SDE 시뮬레이션이 필요했다”는 구분은 과도하다.

**향후 적용**: 이론이 novelty를 지탱하는 논문에서는 필요한 유도를 생략하지 않는다. 다만 유도가 길수록 원문 기호·부호·중간식과 일치하는지 더 엄격히 검수한다.

## VIRAL의 변환 방식

[블로그 글](../_posts/Paper%20Reivew/Multimodal/2025-09-17-VIRAL.md) · [게시 시점 원문 v1](https://arxiv.org/pdf/2509.07979v1)

초점은 **MLLM에 들어간 시각 정보가 내부에서 왜 소실되고, 어느 layer의 어떤 loss로 보존되는가**다. 결과 소개보다 문제 진단 → 단순 재주입 → 실패 → alignment → 더 좋은 teacher라는 인과 흐름을 길게 남긴다.

| 원문 위치 | 포스팅에서의 처리 | 무엇을 남기거나 더 설명했는가 |
| --- | --- | --- |
| §1/3의 전제 | MLLM·Visual Instruction Tuning·Representation Alignment·VFM Background | 모듈 역할과 두 단계 학습, teacher의 의미를 앞에서 설명 |
| §1 | Introduction과 Goal·Motivation·Contributions | caption이 언급하지 않는 개수·색·공간 배치를 유지할 학습 동기가 약하다는 예시 보존 |
| §2 | Related Works 두 계열 보존 | 내부 layer별 정보 흐름과 입력 encoder 개선을 구분하고 자기 방법 위치 설명 |
| §3 Eq. 1/2 | Preliminaries에서 shape와 likelihood 재정의 | `N×D_z`, `N×D`, text 길이, concat과 loss를 기호별로 설명; Background와 일부 중복 |
| §4.1, Fig. 2(e) | 시각 정보 손실 진단 | CKNNA 감소와 중간 layer의 국소 회복을 문제의 근거로 읽음 |
| §4.2 Eq. 3, Fig. 2(b) | post-projection residual 별도 소절 | 기존 text loss를 유지한 채 feature 경로만 보강한다는 통제 설명 |
| §4.2 Eq. 4, Fig. 2(c) | pre-projection residual 실패 보존 | raw feature의 언어 정렬 부족을 최종 방법의 동기로 연결 |
| §4.3 Eq. 5/6, Fig. 2(d) | VRA loss의 projector·target·stop-gradient 설명 | forward feature 전달과 representation 제약의 차이를 풀어 설명 |
| §4.3의 MMVP 결과 | 기존 encoder alignment의 한계 | 좋은 보존만 아니라 encoder 편향까지 전이될 수 있음을 남김 |
| §4.3 Eq. 7, Fig. 3 | From encoder features to other VFMs | 기존 encoder target에서 외부 teacher target으로 바꾸는 과정 설명 |
| §5.1 | 모델·데이터·VFM·평가 기준으로 나눔 | LoRA, 동일 665K 데이터, projector 구성·GPU, 지표 범주 등 비교 조건 보존 |
| §5.2, Table 1 | encoder·LLM 크기·종류의 일반성 | 결과를 여러 backbone과 연결; 세부 대표 숫자는 표에 두는 비중이 큼 |
| §5.3, Table 2 | VFM·단일 layer·multi-layer·loss ablation | teacher, target depth, direct/relation alignment의 설계 선택을 해설 |
| §5.4, Fig. 4 | Attention analysis | 공간적 entropy와 중요한 영역에 집중하는 패턴을 연결 |
| §5.5, Table 3 | Training efficiency | step별 결과를 학습 신호의 이득과 연결 |
| §5.6, Table 4, Appendix A | Robustness analysis | permutation에서 더 크게 악화되는 결과가 공간 정보를 더 사용한다는 증거가 될 수 있음을 해설 |
| §5.7, Fig. 5 | Qualitative Results | PCA 표현과 객체 개수·공간 질문의 답변 차이를 실제 사례로 설명 |
| Conclusion | 별도 limitation 확장 | teacher 비용·편향과 다른 modality의 검증 필요성을 추가 판단으로 정리 |

**그림 선택**: Fig. 2의 세 경로를 개별 crop으로 쓰고 CKNNA 그래프를 따로 확대한다. Fig. 3–5, Table 1–4의 핵심 근거도 포함한다. 부록의 전 layer PCA·VFM 비교·추가 attention 그림은 본문에 모두 옮기지 않는다.

**내용의 배분**: 인코더 자체의 품질보다 내부 representation 보존을 더 오래 설명한다. 실패한 raw residual을 삭제하지 않아 loss 기반 alignment가 필요한 이유를 이해할 수 있게 한다. permutation의 결과를 단순 robustness 성능으로 읽지 않고 공간 구조 사용 여부의 진단으로 읽는 점도 특징이다.

**정확성 주의**: 7B를 “7억”이라고 쓴 부분은 70억으로 구분해야 한다. Table 1의 Qwen/CLIP 설정에는 POPE 감소가 있고 일부 점수는 동률이므로 “모든 설정의 모든 task가 항상 개선”은 표보다 강하다. Table 2에서 DINOv2는 종합적으로 유력한 teacher지만 모든 개별 지표의 최고는 아니다. 1K step부터 모든 지표에서 높게 시작하는 것도 아니다. 글의 MMVP를 video QA 또는 이미지-비디오 평가로 설명한 부분, Depth Anything을 SAM 계열과 섞은 부분, POPE 약어의 임의 풀이는 원문의 평가·모델 목록과 일치하지 않는다. permutation은 encoder 출력의 token 순서를 섞는 실험이므로 픽셀 이미지를 다시 shuffle해 encoder에 넣는 실험과 구별해야 한다.

**향후 적용**: 분석+방법 논문은 관찰과 가설, 통제한 단순 대안, 최종 해결책을 한 흐름으로 보존한다. “더 강한 teacher”의 효과와 “새 loss”의 효과를 비교 조건으로 분리한다.

## CoIBA의 변환 방식

[블로그 글](../_posts/Paper%20Reivew/Computer%20Vision/2025-09-05-CoIBA.md) · [올바른 공식 원문](https://openaccess.thecvf.com/content/CVPR2025/papers/Hong_Comprehensive_Information_Bottleneck_for_Unveiling_Universal_Attribution_to_Interpret_Vision_CVPR_2025_paper.pdf) · [보충자료](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Hong_Comprehensive_Information_Bottleneck_CVPR_2025_supplemental.pdf)

글 안의 arXiv 링크 `2404.02905`는 VAR다. 링크가 가리키는 논문을 CoIBA로 간주하지 않고 제목·저자에 맞는 CVPR 공식 원문을 찾아 대조했다. 올바른 arXiv 식별자는 [2507.04388](https://arxiv.org/abs/2507.04388)이다. 기존 포스팅 링크 자체는 이번 요청에서 수정하지 않았다.

초점은 **한 layer가 놓치는 결정 근거를 여러 layer의 정보 병목으로 모으는 방식과 이론·평가의 의미**다. 최근 표본에서 Background의 수학 설명이 특히 길다.

| 원문 위치 | 포스팅에서의 처리 | 무엇을 남기거나 더 설명했는가 |
| --- | --- | --- |
| §1의 해석 가능성 문제 | ViT와 XAI Background | 블랙박스와 attribution map이 필요한 이유를 현재 task와 연결 |
| IB 전제 | Information Bottleneck Background 확장 | entropy·conditional entropy·mutual information·KL 형태부터 시작 |
| §3.1 Eq. 1/2 | IBA Background와 Method에서 재설명 | lambda의 양 끝값, noise 통계, relevancy/compression의 trade-off를 풀어 설명 |
| §1/3.2, Fig. 1/3/4 | Introduction/Motivation | 단일 최적 layer 부재와 IBA*의 easy/hard 결과를 최종 방법의 동기로 남김 |
| contributions | Goal·Motivation·Contribution | multi-layer, shared damping, upper bound, 설명 평가라는 축으로 재정리 |
| §3.3 Eq. 3 | Information Restriction | sequential bottleneck, `R′`, patch별 lambda와 채널 broadcast, sigmoid 초기값을 해설 |
| §3.3 Eq. 4 | Information Bottleneck Objective | 첫 입력·마지막 출력·층별 압축·평균·beta 역할을 기호별로 설명 |
| §3.3 Eq. 5/6 | KL 소절 | marginal 계산이 어려운 이유와 근사 prior의 상계를 설명 |
| Eq. 4–6의 의미 | 접힌 추가 해설 | 연속된 MI 합의 직관, 구현 관점 요약, relevancy와 압축의 의미를 따로 보충 |
| §3.4 Eq. 7/8 | Variational Upper Bound | 첫 layer의 MI를 통한 단일 beta, 계산 단순화와 theoretical gain을 자세히 풀음 |
| Fig. 2/5 | 이론적 이득과 Overview에 그림 이동 | 기존/제안 경로를 반복 확인하고 전체 알고리즘을 번호 목록으로 회수 |
| §4.1 | 모델·데이터·비교군·최적화 | 각 입력마다 lambda를 최적화하는 protocol까지 설명 |
| §4.2, Fig. 6 | FunnyBirds | completeness·correctness·contrastivity를 독자용 설명으로 확장 |
| §4.3, Table 1 | Insertion/Deletion | 중요한 영역을 제거/복원했을 때의 output 변화와 AUC 방향 해설 |
| §4.4, Table 2 | ROAD | mask leakage 문제와 MoRF/LeRF가 무엇을 확인하는지 설명 |
| §4.5, Fig. 7 | Difficulty-aware 평가 | 전체 평균보다 confidence별 성능과 어려운 분포에 주목 |
| §4.6, Fig. 8/9/10 | layer 범위·shared lambda·upper bound ablation | 이론적 의도가 평가 개선으로 이어지는지 각각 해설 |
| Conclusion | 별도 Limitation 추가 | per-input optimization, randomness, post-hoc 설명의 한계, task 범위를 추가 |

**그림 선택**: 대부분의 본문 방법·평가 증거를 사용한다. Fig. 1의 누적 layer 선택 분포만 crop하고, Fig. 8의 departure/arrival 그림은 따로 실리지 않는다. 글의 layer 범위 ablation 자리에 사용한 `f9.png`는 원문의 Fig. 9인 MI·accuracy·shared lambda 그림이어서 설명과 패널 대응이 어긋난다. Fig. 10(c)는 이미지에 포함되지만 Uniform Channel Perturbation 실험의 독립적인 해설은 생략된다.

**추가 해설**: 수학적 양을 실제 구현·직관으로 세 번 읽어주는 방식이 특징이다. optimizer·초기 lambda·patch별 shape가 들어가 있어 단순 이론 요약에서 실행 절차까지 이어진다. 그러나 이것이 코드 대조를 완료했다는 뜻은 아니다.

**정확성 주의**: IB 상계를 설명할 때 보장 범위를 모델·가정·변분 근사에 한정해야 한다. “모델 입력-출력 MI가 모든 내부 MI보다 크다”는 일반 명제는 성립하지 않으며, Markov chain의 data processing 관점으로 정확히 설명해야 한다. lambda는 정보를 통제하는 변수이고 최종 attribution을 언제나 lambda 값 자체와 동일시할 수 있는지는 산출 정의를 확인해야 한다. FunnyBirds에 대한 “기존 이미지에 bird patch 삽입·사람 box label”과 ROAD의 “대조 클래스 patch로 교체” 설명은 이번 원문에서 뒷받침되지 않는다. Insertion/Deletion AUC를 삭제 후 분류 accuracy 자체로 표기하는 것도 구별해야 한다.

**생략**: 본문의 Uniform Channel Perturbation 검증과 supplement의 sanity·sensitivity·추가 CNN·다양한 설정별 정량 결과는 자세히 다루지 않는다. 글은 비용을 정성적으로 한계에 넣지만, supplement가 보고한 실제 비용 비교 수치는 남기지 않는다.

**향후 적용**: 정보이론 논문은 기호·최적화 목표·근사·이론 가정·실제 attribution 산출을 따로 설명한다. 평가 이름에 설명을 붙일 때는 benchmark 자체의 원문도 확인해야 한다.

## VAR의 변환 방식

[블로그 글](../_posts/Paper%20Reivew/Computer%20Vision/2025-09-05-VAR.md) · [원문](https://arxiv.org/pdf/2404.02905v2)

초점은 **LLM의 AR 성공을 이미지에 옮길 때 무엇을 생성 단위로 삼아야 하는가**다. VQ tokenizer부터 시작하여 next-scale 구조, 효율, scaling, editing으로 이어진다.

| 원문 위치 | 포스팅에서의 처리 | 무엇을 남기거나 더 설명했는가 |
| --- | --- | --- |
| §2.1/3.1 | AR Background | chain rule·prefix, scaling과 zero-shot을 LLM과 연결 |
| §3.1 Eq. 2–5 | Tokenization Background | codebook index, nearest-vector lookup, reconstruction·perceptual·GAN loss 역할을 설명 |
| §3.1의 flattening 한계 | Background와 Motivation | 방향성·공간 구조·효율을 명시; 일부 설명은 반복 |
| §2.2 Related Work | Image generation model로 Background에 통합 | AR·masked·diffusion의 차이를 비교하며 연구 계열의 위치를 잡음 |
| Abstract/Introduction | Goal·Motivation·Contribution | next-scale이라는 생성 단위 전환을 중심에 둠; 공개 code 기여는 짧게 별도 언급 |
| §3.2, Fig. 4 | Method의 두 Stage | tokenizer 학습과 AR Transformer 학습을 분리 |
| Algorithm 1/2 | Stage 1 번호 목록으로 확장 | downsample→quantize→lookup→upsample→residual subtraction, reconstruction의 누적합을 해설 |
| §3.2 Eq. 6, Fig. 2 | Stage 2의 확률 분해 | 전체 token map이 한 AR 단위이며 같은 scale의 token을 병렬 생성한다는 차이 강조 |
| §3.2/4 | mask·cache·class condition | 학습과 inference를 구분하고 기존 AR 한계와의 대응을 목록으로 설명 |
| §4 Eq. 7/8 | 모델/학습 세부 | depth와 width·head·parameter count의 관계를 남김 |
| §5.1, Table 1/2 | 품질·속도·데이터·scaling 비교 | baseline 계열, tokenizer 차이, rejection sampling, 상대시간을 해석 |
| §5.2, Fig. 5 | Power-law Scaling Laws | log-log 선형성과 test loss/error 의미를 배경부터 해설 |
| §5.3, Fig. 7 | 생성 품질 그림 | 모델 크기·학습량 증가의 시각적 효과를 보여줌 |
| §6, Fig. 8 | Zero-shot Task Generalization | in/out-painting·class editing의 teacher forcing과 mask 조작을 풀어 설명 |
| §7, Table 3 | 구성요소별 ablation | paradigm 자체, AdaLN, top-k, CFG, QK norm, scale-up을 각각 설명 |
| §8/9 | Limitations/Future Work와 Conclusion | tokenizer·text·video 확장의 미검증 범위를 남기며 결론에서 다시 요약 |

**그림 선택**: 생성 예시, 세 방식 비교, 두 단계 방법 그림, 알고리즘, 품질·속도 표, parameter scaling, 질적 scaling, editing, ablation을 사용한다. Fig. 6의 compute scaling 그래프와 Appendix A의 token dependency heatmap은 보여주지 않는다. Appendix B의 복잡도 증명 전체도 옮기지 않는다.

**설명 분배**: tokenizer를 사전 지식으로 치지 않고 feature→index→embedding→image를 다시 설명한다. 알고리즘을 줄마다 읽고, loss·mask·class condition을 이해 가능한 구조로 풀어낸다. 원문에서 별도 implementation details였던 설정은 Stage 설명 뒤로 합친다.

**축약**: 실제 power-law fitting 식과 compute frontier는 대부분 추세로 압축한다. 대신 parameter 수와 병렬 생성의 의미, 편집 응용, ablation에는 많은 설명을 쓴다.

**정확성 주의**: “token에 양방향 상관이 있으므로 chain rule/AR 자체가 수학적으로 불가능”하다는 독립적 사실로 확대하면 안 된다. 이는 원문 저자의 모델링 동기와 일반 확률분해의 유효성을 구분해야 하는 지점이다. `n`은 픽셀 해상도가 아니라 **latent token grid**의 크기이며, 글의 “256²에서 약 65k 대 5 step”은 표의 VQGAN 256 step·VAR 10 step과 맞지 않는다. `O(n⁶)`와 `O(n⁴)` 비교도 원문의 계산·scale 가정을 명시해야 한다. final residual이 정확히 0이 된다고 보장하지 않는다. Queue reconstruction을 큰 map부터 pop한다고 쓰거나 convolution의 kernel을 원문 확인 없이 지정하지 않는다. 18.5M은 1.85억이 아니다. Top-k와 nucleus/top-p sampling도 구별한다.

**향후 적용**: AR 생성 논문에서는 “모델이 무엇을 한 번에 예측하는가”, tokenization과 generation의 차이, 조건부 mask, 단계 수·복잡도·실시간을 우선 설명한다. class-conditional 결과를 일반 T2I superiority로 확대하지 않는다.

## SigLIP의 변환 방식

[블로그 글](../_posts/Paper%20Reivew/Multimodal/2025-09-02-SigLIP.md) · [원문](https://arxiv.org/pdf/2303.15343v4)

초점은 **loss의 작은 변화가 batch·분산 통신·메모리·자원 제약을 어떻게 바꾸는가**다. 수식과 코드의 대응을 매우 자세히 쓰고, result를 핵심 결론 위주로 선택한다.

| 원문 위치 | 포스팅에서의 처리 | 무엇을 남기거나 더 설명했는가 |
| --- | --- | --- |
| Introduction/Related Work | Contrastive Learning Background | positive/negative, CLIP의 large batch와 all-gather 병목을 앞에 둠; MoCo·SimSiam을 추가 사례로 설명 |
| Abstract/Introduction, Table 1 | Goal·Motivation·Contribution | 작은 자원·작은 batch 성능과 큰 batch 포화라는 핵심 질문 중심 |
| §3.1 | 두 방향 softmax loss를 별도 식으로 분리 | encoder normalization, 분모 경쟁, 양방향 평균과 계산 의미를 설명 |
| §3.2 | Sigmoid binary classification tutorial | positive/negative 예시, bias, temperature, label과 loss의 관계를 기호별로 풀이 |
| Algorithm 1 | code block과 줄별 해설 | normalization→logit→label sign→log-sigmoid→합산의 연산 대응 |
| §3.3, Fig. 1 | chunked implementation | device 3개·local batch 4개 예시를 순서 목록으로 읽어줌 |
| §4.1–4.3, Fig. 2 | batch sweep를 한 결과 장으로 묶음 | SigLiT/SigLIP/mSigLIP 세 panel을 구분하고 32k 근방 포화 해석 |
| §4.4/4.5, Table 1 | 적은 TPU 결과 중심 | frozen pretrained와 from-scratch 효율을 비교하려는 의도; 세부 fine-tuning recipe는 뒤에서 한계로 이동 |
| §4.6, Table 3 | 최종 공개 모델 비교 | 분류뿐 아니라 retrieval 성능도 정리 |
| §4.10, Fig. 7 | label noise robustness | 다섯 corruption 종류와 그래프 추세를 자세히 해설 |
| §4.7 | Limitation으로 이동 | 큰 batch에서 beta2를 바꿔야 하는 안정성 문제 |
| §4.8 | Limitation으로 압축 | random/easy/hard negative의 효과와 mining 문제 |
| §4.9 | Method와 Limitation에 분산 | bias 필요성은 자세히 설명하지만 초기값 ablation 표는 보여주지 않음 |
| §4.5 | Limitation으로 이동 | pretrained image tower의 weight decay 조정 필요성 |
| Conclusion | 성능·효율·noise를 묶음 | 방법을 실용적인 recipe로 이해한 결론 |

**그림 선택**: Fig. 1/2/7, Table 1/3을 사용한다. 같은 Table 1 이미지를 introduction과 resource result에 두 번 참조한다. Fig. 3/4/5/6, Table 2/4는 본문 이미지로 보여주지 않는다. 다국어 bottleneck embedding 설계는 자세히 설명하지 않는다.

**추가 해설**: pair별 label의 부호가 positive와 negative loss를 한 줄로 통일하는 방법, loss 누적이 정확히 모든 pair를 한 번씩 포함하는 방법, local 청크 크기가 peak memory를 줄이는 방법을 원문보다 단계적으로 설명한다. “분산 구현”은 주변 구현 정보가 아니라 방법의 작동 이유로 취급한다.

**정확성 주의**: 앞의 label `0/1` 설명과 뒤의 signed formula `+1/−1`을 혼용하면 loss가 달라진다. Algorithm 1에 맞춘 일관된 표기는 `s_ij = t〈x_i,y_j〉 + b`, `L = -(1/N)Σ_ij log σ(z_ij s_ij)`이며 signed label을 사용한다. 원문의 식 표기와 코드까지 대조하여 부호를 확인해야 한다. 초기값은 `t′=log(10)`, `t=10`, `b=-10`이지 `t′=10`, `t=exp(10)`이 아니다. 원문의 4-chip 작은 frozen image tower는 B/8인데 글에서 B/16이라고 설명한다. 서로 다른 pretrained/from-scratch 자원 결과를 loss만 바꾼 공정한 비교로 읽으면 안 된다. 작은 batch의 장점을 작은 dataset으로도 충분하다는 결과로 확대할 수 없다.

**향후 적용**: loss 중심 논문은 formula·기호·초기값·코드·분산 구현을 일치시킨다. 메모리 절감과 전체 pair 연산량의 제거는 구분한다. 효율 주장은 하드웨어·학습량·pretrained 상태와 함께 제시한다.

## 공통적으로 남기고 추가하는 정보

| 정보 | 실제 표본의 공통 처리 | 향후 기준 |
| --- | --- | --- |
| 연구 문제 | 실패 현상과 task를 먼저 정의 | 기존 방식의 문제를 구체적으로 남김 |
| 선행연구 | 구현 단계·목표·한계로 계열화 | 이름·citation만 나열하지 않음 |
| novelty | loss·representation·연산 단위의 변화 | backbone 이름보다 무엇을 바꾸었는지 먼저 설명 |
| 학습 단계 | stage별 데이터·모듈 역할 설명 | training/inference·frozen/trainable·gradient 구분 필수 |
| 수식 | 주요 식과 기호 유지, 필요한 유도 추가 | 원문과 일치하는 기호·차원·부호·조건을 검수 |
| 직관 | 쉬운 재진술, 경로, 예시, 짧은 코드 | 사실·단순화·가설을 구별 |
| 그림 | 원문 figure crop을 관련 문단 옆에 배치 | 패널·색·축을 실제로 읽어줌 |
| 표 | 전체 표는 이미지, 핵심 결과만 문장 | 예외·설정 차이·sampling 조건도 남김 |
| ablation | 요소별 장으로 풀고 설계 선택을 해석 | 어느 가설의 근거인지 연결 |
| 한계 | 원문의 limitation 외에 구조상 제약 추가 | 원문 주장과 추가 판단을 구분 |
| 비용 | 효율성 관심은 강하지만 숫자가 생략되기도 함 | 효율 주장의 필수 조건으로 실제 비용과 포함 범위 보완 |
| 개인적 맥락 | 관심 있는 문제에 설명 분량을 배분 | 배경 경험은 사용자 제공 사실만 사용 |

## 배경지식과 분야 취향

아래는 글에서 관찰한 내용과 해석을 구분한 것이다. 사용자의 능력이나 취향 전부를 평가한 결과는 아니다.

| 근거 | 관찰 또는 추론 | 작성에 반영하는 방법 |
| --- | --- | --- |
| GAN·DDPM·DDIM·score·flow 관련 글 제목과 다수 수식 | 생성 모델의 확률적 구조를 계속 연결해서 읽는 경향 | 새 생성 방법을 기존 loss·sampling·trajectory와 연결 |
| 최신 Diffusion Bridge·VLV·VIRAL·SigLIP | image/text embedding과 그 안의 세부 정보 보존에 반복적으로 주목 | latent 의미·정렬 방향·정보 손실을 중심 질문으로 삼음 |
| VLV의 3D awareness, VIRAL의 spatial/counting, CoIBA attribution | 점수뿐 아니라 공간·세부·내부 동작을 이해하려는 경향 | qualitative·내부 representation·진단 실험을 생략하지 않음 |
| stage·teacher·frozen·모듈 재사용에 긴 설명 | pretrained 모델 결합과 작은 변경의 효과에 관심이 있어 보임 | 모듈 상태와 학습 신호를 구체적으로 설명 |
| SD3 유도, CoIBA IB/KL, SigLIP 줄별 코드 | 수학과 구현을 연결해 이해하려는 설명 방식 | 공식만 제시하지 않고 중간 단계와 연산 의미를 남김 |
| DINO 추가 질문, ResCLIP Code 장 | 원문 바깥의 의문과 구현 확인도 학습 기록에 포함 | 필요할 때 질문·code review를 선택적으로 추가 |
| UniReal의 개인적 느낌 장 | 사용자가 VLM domain generalization용 합성 데이터, reference 객체의 상황·camera view 변화를 다룬 경험을 직접 언급 | 관련 글에서 연결할 수 있지만 새 연구 경험·성과·감상은 만들지 않음 |

따라서 입문자 수준으로 모든 개념을 처음부터 설명할 필요는 없지만, 익숙한 모델의 이름을 안다고 핵심 유도나 단계의 차이까지 생략할 수는 없다. 실제로 최근 글은 기본 MLLM/contrastive 개념도 방법 이해에 필요한 만큼 다시 설명한다.

## 문체와 편집에서 유지할 요소

1. **기술 영어와 한국어 설명의 혼합**: 용어를 모두 어색하게 번역하지 않고 기능과 문맥을 한국어로 설명한다.
2. **짧은 설명 단위**: 정의·변형·해석·결과를 별도 문단으로 나누어 다시 읽기 쉽게 한다.
3. **재진술의 역할**: `즉`, `다시 말해`, `쉽게 말해` 뒤에는 직관이나 적용 결과가 나온다. 같은 말을 무한히 반복하는 형식으로 모방하지 않는다.
4. **강조와 인용문**: bold는 핵심 개념·변화·결과, quote는 서지정보와 보충 해설·예시·주의점에 사용한다.
5. **목록과 순서**: 모듈 역할은 bullet, 학습·tokenization·분산 절차는 번호로 정리한다.
6. **접힌 설명**: SD3 두 개, CoIBA 한 개의 details는 유도·비교·직관을 본문 근처에 보관한다. 선택적 깊이를 제공하는 편집이다.
7. **논문별 가변 구조**: Method라는 제목 유무, related work 위치, limitations와 추가 질문의 위치가 다르다. 내용의 논리에 맞추는 것을 공통 스타일로 본다.
8. **개인적 학습 연결**: 기존 글 링크·비교·질문·코드로 자신의 학습 맥락을 이어간다. 개인 감정은 모든 글에 필수적으로 들어가지 않는다.

## 스타일로 복제하지 않을 문제

아래는 기존 글을 고치라는 요청이 아니라, 이후 작성에서 같은 오류가 생기지 않도록 분리한 점검 기록이다.

| 문제 유형 | 확인한 예 | 향후 처리 |
| --- | --- | --- |
| 잘못된 source 연결 | CoIBA 링크가 VAR로 연결 | 제목·저자·식별자 일치부터 확인 |
| 복사 잔재 | SD3 서지 quote의 VIRAL 제목, Diffusion Bridge 링크 앞 `ßß` | front matter·H1·서지·링크 전체 검수 |
| 단위 오류 | SD3 8B→8억, VIRAL 7B→7억, VAR 18.5M→1.85억 | 원래 M/B를 보존하고 환산 확인 |
| 수식과 코드 불일치 | SigLIP signed label과 0/1, log-temperature 초기값 | 한 convention으로 formula·코드·주석 통일 |
| 수식 전사 오류 | SD3 RF weighting 부호, FM marginal/conditional 혼동, placeholder `ㅊ` | 원문 식과 각 중간식 대조 |
| pipeline 혼동 | VLV caption condition과 VAE/noised latent | 두 경로와 학습 파라미터를 분리 |
| 평가 정의 확대 | CoIBA AUC를 accuracy로 설명, VIRAL MMVP/video 설명 | benchmark·지표의 실제 정의 확인 |
| 결과의 예외 누락 | VIRAL 모든 task/초기 step에서 개선이라고 일반화 | 동률·감소·설정별 최고가 다른 결과도 남김 |
| 사실보다 강한 결론 | 상관을 완전한 정보 보존 증명, scaling을 비례 증가로 표현 | 관찰 범위·가정·저자 해석을 구분 |
| 그림과 설명의 불일치 | CoIBA layer 범위 설명에서 Fig. 9 패널 사용 | 원문 번호뿐 아니라 패널 내용 대조 |
| 근거 없는 세부 보충 | FunnyBirds/ROAD 절의 상세 작동을 원문 확인 없이 확장 | 필요한 별도 1차 출처 확인 후 설명 |
| front matter 형식 | CoIBA `use_math: true. TAG names…`, 날짜 `+/-0000` | 새 글에서 유효 boolean·timezone 사용 |
| 용어 오기 | Methology, Enxoders, Alignmnet 등 | 영어 용어·heading spelling 검수 |

이 분석이 요구하는 충실함은 원문의 모든 문장을 복사하는 것이 아니다. 논문의 문제, 새 설계가 필요한 이유, 실제 작동과 평가의 연결이 끊기지 않도록 정보의 중요도를 판단하고, 작성자에게 필요한 설명을 더하는 방식이다.
