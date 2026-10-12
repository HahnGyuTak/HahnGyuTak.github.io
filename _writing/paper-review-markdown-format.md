# 논문 리뷰 내용별 Markdown 작성 명세

이 문서는 **어떤 내용을 어떤 Markdown 문법으로, 어디에 배치하는지** 정한다. 논문 선택·설명 깊이·문체는 [작성 기준](paper-review-style.md), 논문을 글로 재구성한 근거는 [원문 대조 분석](paper-review-analysis.md)과 함께 적용한다. 내용이 맞더라도 제목 계층, 서지정보, 수식, 목록, 토글의 작성 형식이 맞지 않으면 이 블로그의 작성 기준을 충족한 것으로 보지 않는다.

2026년 10월 12일 전체 39편의 본문 문법을 다시 조사했고, 최신 7편의 실제 사용을 기본으로 삼았다. 사용자가 직접 지정한 H1 제목, 서지 인용문, Background부터 시작하는 섹션 흐름, 설명용 H3, dollar 수식과 기호 목록, 긴 부연의 HTML 토글은 **앞으로의 필수 규칙**이다. 과거 글에 다른 표기가 있어도 이 규칙을 우선한다.

## 내용과 문법의 대응

| 내용의 역할 | 사용할 문법 | 배치와 사용 기준 |
| --- | --- | --- |
| 게시물 메타데이터 | 파일 첫머리의 YAML `---` 블록 | title, date, categories, tags, use_math, typora-root-url을 기록 |
| 포스팅 저장 디렉토리 | `_posts/Paper Reivew/분야/` 아래 실제 Markdown 파일 | 핵심 주제로 분류하고 적합한 디렉토리가 없으면 생성; categories·tags·상대 루트 경로도 함께 설정 |
| 본문 블로그 제목 | `# **[논문 리뷰] 영문 논문 제목**` | front matter 다음, 본문 H1은 하나 |
| 논문 세부정보 | 연속된 `>` 인용문 블록 | 제목 → 저자 → 학회·연도 → 관련 링크 순서 |
| 서지정보 중 논문 제목 | `> **논문 제목**` | 서지 블록 첫 항목, 본문 H1의 논문명과 일치 |
| 논문·코드·프로젝트 링크 | `> [[arXiv](URL)] [[Github](URL)]` | 서지 블록 마지막 행, 실제 존재하는 링크만 포함 |
| 주요 섹션 | `## **섹션명**` | Background부터 시작하는 유사한 흐름, 원문의 모든 절을 자동 복사하지 않음 |
| 배경 개념·학습 단계·모듈·실험 질문 | `### **소절명**` | 해당 H2 아래, 원문에 없어도 이해에 필요하면 추가 |
| H3 안의 독립적인 세부 항목 | `#### **항목명**` | 예: Decoder Training 아래 augmentation과 입력 adapter |
| H4 아래의 더 작은 기술 요소 | `##### **요소명**` | 실제 상하위 관계가 있을 때만 사용 |
| 짧은 문단 안의 구분 라벨 | `**목적**`, `**주요 결과**` | 목차에 올릴 독립 소절이 아닐 때 사용 |
| 일반 설명·결과 해석 | 평문 문단 | 문단 사이에 빈 줄, 한 문단에서 하나의 설명 단계를 전개 |
| 핵심 개념·설계 차이·중요 결과 | `**굵은 강조**` | 핵심 구절 위주, 문단 전체를 상시 강조하지 않음 |
| 보조적인 강조·관례적 학술 표기 | `*기울임*` | 제한적으로 사용; 핵심 강조는 bold가 기본 |
| 과거 메모의 삭제·취소 표시 | `~~취소선~~` | 일부 초기 글의 개인 메모에 사용; 핵심 기술 설명의 기본 강조로 쓰지 않음 |
| 모듈 역할·조건·비교 항목·한계 | `- 항목` 또는 `* 항목` | 순서가 없는 병렬 정보, 한 목록 안에서는 marker를 통일 |
| 학습·추론·알고리즘의 순서 | `1.`, `2.`, `3.` | 실행 순서가 있는 단계, 필요하면 단계 아래 세부 bullet |
| 수식의 주요 기호 설명 | `- $기호$: 설명` | 해당 수식 바로 아래에 목록으로 기입; 새롭거나 중요한 기호를 생략하지 않음 |
| 문장 안의 기호·짧은 관계식 | `$...$` | 설명 문장 안에 포함; `\(...\)`로 대체하지 않음 |
| 독립된 핵심 수식 | `$$` 블록 | 앞에 목적을 설명하고 아래에 기호·해석 배치; `\[...\]`로 대체하지 않음 |
| 여러 줄의 식 유도 | `$$` 안의 `aligned` 또는 `align` | 필요한 변형을 줄로 나누고 대응 지점을 정렬 |
| 원문의 식 번호 | 수식 내부의 `\tag{번호}` | 원문 번호와 실제 식이 일치할 때만 사용 |
| 짧은 직관·예시·주의점 | 본문의 `>` 인용문 | 관련 설명 바로 옆; 서지 블록과 역할을 구별 |
| 짧은 부연의 하위 부연 | `> >` | 기존 중첩 인용 사용을 따르되 깊은 중첩을 계속 늘리지 않음 |
| 긴 추가 유도·비교·질문 답변 | HTML `<details><summary>…</summary>` | 관련 본문 다음에서 접기; 내부에 `<div markdown="1">` 사용 |
| 코드 변수·함수·설정 이름 | 백틱으로 감싼 inline code | 수학 기호와 실제 코드 식별자를 구별 |
| 여러 줄 코드·의사코드 | fenced code block | 가능한 경우 언어 표시; 코드 아래에 연산 의미 설명 |
| 논문 그림·결과 표 이미지 | `![설명](/assets/img/...)` | 관련 방법·실험 옆에 배치하고 무엇을 볼지 설명 |
| 폭을 조절할 그림 | HTML `<img ...>` | 단순 Markdown 이미지로 충분하지 않을 때 사용 |
| 나란히 볼 두 그림 | HTML flex `<div>`와 두 `<img>` | 실제 비교에 필요한 경우만 사용 |
| 작성자가 재구성한 비교 표 | Markdown pipe table | DINO 등의 개념·방법 비교에 사용; 논문 결과 표 이미지와 구별 |
| 본문 참고·기존 글·구현 링크 | `[문맥에 맞는 이름](URL)` | 해당 설명 근처, 불필요한 독립 URL 나열을 피함 |
| 선택적인 시각적 간격 | `<br>` | 토글 뒤나 큰 설명 전환 등 필요한 곳에 제한적으로 사용 |
| 큰 설명 묶음의 구분 | 본문의 `---` | 일부 기존 글에서 사용; 모든 H2 아래 장식으로 반복하지 않음 |

`-`와 `*`는 기존 최신 글에서도 모두 사용한다. 다만 **수식 아래 기호 설명은 사용자가 지정한 `-` 형식으로 통일한다.** Heading을 만들지 않아도 되는 작은 라벨까지 H3로 바꾸지 않고, 독립적인 설명 단위는 bold 라벨만으로 숨기지 않는다.

## 제목과 섹션 계층

### 본문 제목과 메타데이터 제목

front matter의 `title:`은 게시물 메타데이터다. 본문 제목을 대신하지 않는다.

새 글의 주제별 저장 디렉토리는 [작성 기준의 분류 규칙](paper-review-style.md#주제별-디렉토리-선택과-생성)을 따른다. 아래 예시의 `typora-root-url: ../../../`는 기본 분야 폴더 기준이며, 새 하위 디렉토리에 넣으면 실제 경로 깊이로 다시 계산한다.

```yaml
---
title: "[Paper Review] 논문 약칭"
date: YYYY-MM-DD HH:MM:SS +0900
categories: [Paper Review, 분야]
tags: [핵심주제, 방법]
use_math: true
typora-root-url: ../../../
---
```

```markdown
# **[논문 리뷰] 정확한 영문 논문 제목**
```

최근 글은 약칭을 metadata title에, 논문 전체 제목을 H1에 쓰는 경우가 많다. 둘을 같은 문자열로 강제하지 않지만 서로 다른 논문명으로 남기지는 않는다. H1은 글 전체를 대표하며 각 주요 장을 H1로 작성하지 않는다.

### H2는 글의 큰 흐름

기본 순서는 다음과 같다.

```markdown
## **Background**
## **Introduction**
## **Related Work**
## **Method**
## **Experiments**
## **Conclusion**
```

실제 파일에서는 각 heading 사이에 내용과 빈 줄을 넣는다. 위 예시는 순서만 보여준다.

- Background는 논문의 독립 섹션 여부와 무관하게 이해에 필요한 선행 개념을 모아 시작한다.
- Introduction에서는 문제·아이디어를 서술한 뒤 Goal, Motivation, Contributions를 보통 H3로 나눈다.
- Related Work, Preliminaries는 내용상 필요할 때 넣거나 다른 장에 합친다. 두 장이 반드시 모두 있어야 하는 것은 아니다.
- SD3처럼 수학·architecture가 긴 경우 Method 대신 핵심 방법 이름의 H2를 여러 개 둔다.
- Experiments는 Empirical Results 등 논문과 설명 흐름에 맞는 이름으로 바꿀 수 있다.
- Limitations는 Conclusion 아래 H3 또는 별도 H2로 둘 수 있다. 독립적인 설명 분량과 기존 인접 글의 구성에 맞춘다.
- 원문의 Abstract, 모든 Related Work 소절, References, Acknowledgements, 모든 Appendix를 H2로 복사하지 않는다. 보존해야 할 정보와 읽기 순서를 먼저 결정한다.

### H3는 이해를 위한 소절

원문의 소절과 일대일 대응할 필요가 없다. 다음은 H3를 새로 만들 만한 설명 단위다.

- Background 아래의 VAE, Contrastive Learning, MLLMs 등 개념
- Introduction 아래의 Goal, Motivation, Contributions
- Method 아래의 Pipeline Overview, Stage 1, Stage 2, Inference
- 원문 구현 상세에서 방법 설명으로 옮긴 projector·prefix·soft prompt의 역할
- Experiments 아래의 서로 다른 실험 질문, benchmark 결과, ablation 묶음

```markdown
## **Method**

### **Pipeline Overview**

전체 파이프라인을 먼저 설명한다.

### **Stage 1: Representation Learning**

첫 단계의 입력과 학습 목표를 설명한다.

### **Stage 2: Caption Decoding**

두 번째 단계의 데이터와 decoder를 설명한다.
```

### H4와 H5는 실제 하위 관계

```markdown
## **Method**

### **Decoder Training**

#### **Diffusion-Augmented Text Embeddings**

증강 방법을 설명한다.

#### **Decoder 입력 어댑터**

입력 형태를 바꾸는 모듈을 설명한다.
```

SD3에는 `## Experiments → ### Training at Scale → #### Fine tuning on high resolution → ##### QK-Normalization` 계층이 있다. 깊은 제목을 쓸 수 있지만 글씨 크기만 줄이기 위해 H2에서 H5로 건너뛰지 않는다. 기존 SD3의 RF 항목처럼 계층을 건너뛴 사례는 관찰 기록으로 남기고 새 글의 기본 규칙으로 삼지 않는다.

최신 글도 `### Goal`처럼 bold를 생략한 변형이 있다. 새 글은 `### **Goal**`처럼 제목 강조를 통일한다. 과거의 `## ✏️...`, 장 번호 접두사, 오탈자는 필수 형식으로 복제하지 않는다.

## 논문 세부정보 인용문

H1 다음에 다음 순서로 쓴다. 항목 사이의 빈 줄도 인용문에 속하도록 `>`를 유지한다.

```markdown
> **정확한 영문 논문 제목**
>
> First Author et al.
>
> 확인된 학회와 연도 또는 arXiv preprint
>
> [[arXiv](논문_URL)] [[Github](공식코드_URL)] [[Project Page](프로젝트_URL)]
```

- 제목은 bold, 저자·학회·링크는 각각 별도의 인용문 행에 배치한다.
- 링크의 바깥 대괄호까지 남기는 `[[표시명](URL)]` 형태는 실제 최신 글의 서지 형식이다.
- CVPR 공식 페이지처럼 arXiv가 아니면 `[[Paper](URL)]`로 표시한다.
- 공개 코드나 프로젝트가 없으면 해당 항목을 빼고, 있는 것처럼 placeholder를 남기지 않는다.
- 저자 전체 이름, `et al.`, 저자 이름의 개별 링크, `*et al.*` 표기는 기존 글에 모두 변형이 있다. 저자 수와 원문에 맞게 선택하되 서지 항목 순서는 유지한다.
- 학회 미확인 상태에서 학회·연도를 지어내거나 비어 있는 `>` 행만 정보처럼 남기지 않는다.
- 서지 블록의 `>`는 직접 인용을 뜻하기보다 이 블로그의 정보 블록 형식이다.

## 수식과 기호 목록

### 독립 수식

핵심 식은 목적을 설명한 다음 독립된 dollar 블록으로 쓴다. 여는 `$$`와 닫는 `$$`는 각각 독립 행에 둔다.

```markdown
다음 식은 bottleneck이 원래 feature와 noise를 얼마나 섞을지 나타낸다.

$$
Z = \lambda R + (1-\lambda)\epsilon
$$

- $Z$: bottleneck을 통과한 representation
- $R$: 원래 feature representation
- $\lambda$: 원래 정보를 통과시키는 비율
- $\epsilon$: noise

$\lambda$가 1이면 원래 정보를 유지하고, 0이면 noise만 통과시킨다.
```

- 수식 아래에는 **새롭거나 중요한 기호를 `- $기호$: 설명` 목록으로 반드시 적는다.** 정의·shape·범위·분포·학습 여부가 이해에 필요하면 함께 적는다.
- 바로 앞에서 정의한 기호를 다음 식에서 전부 다시 나열할 필요는 없지만 새로운 기호를 정의 없이 추가하지 않는다.
- 수식을 먼저 설명 없이 나열하고 글 마지막에 기호를 한꺼번에 모으지 않는다.
- 기존 Diffusion Bridge는 Eq. 1–5를 한 묶음으로 제시한 뒤 공통 기호 목록을 둔다. 같은 전제의 연속 식은 이처럼 **식 묶음 바로 아래**에 공통 설명을 둘 수 있다.
- 과거 글에 `$$식$$`을 같은 행에 둔 사례도 있다. 새 글은 위의 독립 행 형식을 기본으로 한다.
- `\[...\]`와 `\(...\)`는 새 글의 math delimiter로 사용하지 않는다. 과거 SAGAN의 bracket 수식은 현재 규칙의 예외 사례다.

### 인라인 수식

```markdown
이미지 feature는 $z \in \mathbb{R}^{N\times D}$이며, $N$은 token 수다.
```

문장 속 기호를 code backtick으로 쓰지 않는다. 실제 코드의 `hidden_states`나 `lambda_weight` 같은 이름은 backtick, 수학의 $z$나 $\lambda$는 dollar다.

### 식 번호와 여러 줄 유도

```latex
$$
\begin{aligned}
z_t &= a_t x_0 + b_t\epsilon \\
\frac{d z_t}{dt} &= a'_t x_0 + b'_t\epsilon
\end{aligned}
$$
```

원문 식 번호를 보존할 때는 해당 식에 `\tag{6}`처럼 번호를 넣는다. 직접 추가한 식에 원문의 번호를 임의로 부여하지 않는다. SD3는 `align`, `\notag`, `\tag`를 함께 사용해 유도 단계와 원문 번호를 구별한다. `&`는 식의 정렬점, `\\`는 수식 내부의 줄바꿈이다. Markdown 문단 줄바꿈과 혼동하지 않는다.

### 인용문 안의 수식

짧은 설명에 식이 포함되면 수식의 모든 행에도 `>`를 붙여 같은 인용문에 둔다.

```markdown
> **짧은 보충 직관**
>
> $$
> z^{vision} = z^{text} + \epsilon
> $$
>
> - $z^{vision}$: vision embedding
> - $z^{text}$: text embedding
> - $\epsilon$: 두 representation 차이를 나타내는 noise 근사
```

식이 중첩 인용문이나 목록 안에 있으면 그 컨테이너의 prefix·들여쓰기를 유지한다. prefix를 잃은 식이 다른 문맥의 독립 수식처럼 떨어지지 않게 한다.

## 본문과 보충 설명의 구분

### 평문과 짧은 구분 라벨

```markdown
이 실험은 정렬된 embedding이 실제 caption 품질로 이어지는지 확인한다.

**실험 개요**

- **목적:** 방법이 보존하는 정보를 확인한다.
- **설정:** 같은 데이터에서 정렬 loss의 유무만 바꾼다.

**주요 결과**

결과와 예외를 설명한다.
```

`**실험 개요**`는 짧은 문단 라벨이며 H3가 아니다. 그 안에 독립적인 여러 방법·실험을 펼쳐야 한다면 H3/H4를 사용한다. 최신 VLV는 이 라벨과 목록을 실험별로 사용하고, Diffusion Bridge는 목적을 평문 첫 문장으로 쓴다. 두 방식 모두 유지 가능한 변형이다.

### 짧은 부연은 인용문

```markdown
> **왜 normalization을 하는가**
>
> embedding의 크기를 통제하여 signal과 noise의 상대 크기를 다루기 쉽게 만든다.
```

인용문에는 본문의 직관, 예시, 해석상의 주의점, 작은 추가 정의를 둔다. 원문의 직접 인용을 옮긴 경우와 작성자가 추가한 해설을 문장으로 구별한다. `> >`는 RMSNorm처럼 보충 안의 짧은 추가 정의에 사용된 사례가 있다. 길어지면 계속 중첩하지 않고 토글로 바꾼다.

### 긴 부연은 HTML 토글

긴 추가 유도·기존 방법과의 상세 비교·옆길로 빠지는 배경·질문 답변은 본문에 길게 펼치지 않고 아래 형식으로 접는다.

```html
<details><summary>FM loss와 CFM loss는 왜 같은 최적화를 주는가</summary>
  <div markdown="1">

이곳에 추가 설명을 작성한다.

필요한 Markdown 목록, 수식, 코드 또는 비교 표를 넣는다.

  </div>
</details>

<br>
```

- `<summary>`에는 실제 설명 주제나 질문을 쓴다. 무의미한 `Description`만 붙이는 것보다 열기 전에 내용을 알 수 있게 한다.
- `<div markdown="1">`를 유지하고 여는 div 다음과 닫는 div 앞에 빈 줄을 둔다.
- 내부 콘텐츠는 실제 포스팅에서 Markdown으로 작성한다. 토글 전체를 fenced code로 감싸면 HTML 소스가 그대로 표시되므로 그렇게 쓰지 않는다. 위 code block은 명세를 보여주기 위한 예시다.
- 닫는 태그는 `</div>` 다음 `</details>` 순서로 쓴다. `<br>`는 토글 뒤 간격이 필요한 경우에 붙인다.
- 본문에는 방법의 핵심, 수식의 의미, 적용 조건을 남긴다. 토글은 그것을 추가로 이해하기 위한 긴 설명을 담는다.
- H3는 해당 설명을 **목차의 독립 단위로 나누는 문법**, 토글은 그 소절 안의 **긴 선택적 설명을 접는 문법**이다. 둘을 구별한다.
- 과거 글에는 `<b>`가 들어간 summary, 한 행에 div까지 이어 쓴 형태, HTML `<ul>/<li>`도 있다. 최신의 `details + div markdown="1"`을 새 글의 기본으로 한다.

## 목록과 들여쓰기

- **순서 없는 병렬 정보**: `-` 또는 `*`
- **순서 있는 실행 과정**: `1.`, `2.`, `3.`
- **수식 기호 정의**: `- $기호$: 설명`
- **단계 아래의 상세 이유·조건**: 한 단계 들여쓴 bullet 또는 짧은 quote

```markdown
1. vision encoder로 feature를 추출한다.

   - 입력: 이미지
   - 출력: patch feature

2. projector로 embedding 차원을 바꾼다.

   > 두 모델의 hidden dimension이 다르기 때문에 필요한 단계다.

3. decoder로 caption을 생성한다.
```

순서 없는 목록의 하위 bullet은 부모보다 들여쓴다. 번호 목록 안의 여러 문단·수식·그림은 번호 다음 본문이 시작하는 열에 맞춰 들여쓴다. 같은 항목의 보충 내용이 바깥 문단으로 떨어지지 않게 한다. 불필요한 탭이나 최상위 본문의 4칸 들여쓰기로 의도하지 않은 code block을 만들지 않는다.

같은 목록 안에서 `-`와 `*`를 무작위로 바꾸거나 `1)`와 `2.`를 섞지 않는다. 번호가 붙은 기여 목록 안에는 핵심 변화와 그 설명을 하위 bullet로 둘 수 있다. 실제 최신 글은 marker가 혼용되기도 하지만 같은 목록의 일관성은 새 글에서 검수한다.

## 그림과 표의 소스 형식

### 원문 figure와 table의 선택부터 삽입까지

논문 리뷰를 작성할 때 필요한 그림·표의 준비도 함께 수행한다. 사용자가 별도로 screenshot을 만들거나 asset 경로를 채워줄 것을 전제로 하지 않는다.

1. **선택**: 전체 아이디어·pipeline, 핵심 모듈, 주요 결과, 중요한 ablation 중 현재 글의 설명에 필요한 figure·table을 고른다. 논문의 모든 이미지를 기계적으로 넣지는 않는다.
2. **원문 대응 기록**: 작업 중 논문 버전, PDF 페이지, Figure/Table 번호, 사용한 panel과 사용할 소절을 연결한다. 보충자료의 이미지는 본문 자료와 출처를 구분한다.
3. **캡처·추출**: PDF 페이지를 충분한 해상도로 렌더링해 필요한 영역을 crop한다. 같은 버전의 TeX source에 원본 이미지가 있으면 PDF와 일치하는지 확인해 활용할 수 있다. 표는 PDF의 실제 결과 표를 캡처한다.
4. **영역 확인**: 핵심 panel, 축·단위·범례, 행·열 제목과 숫자를 보존한다. 원문 캡션을 이미지에 포함할지 본문 설명으로 옮길지 정하고, 캡션이 없어도 어떤 Figure/Table인지 설명에서 알 수 있게 한다. 방법 그림은 단계별 crop을 따로 저장할 수 있다.
5. **저장**: `assets/img/YYYY-MM-DD-주제/`에 PNG 등의 실제 파일로 저장한다. `fig-02-pipeline.png`, `fig-02-stage1.png`, `table-01-main-results.png`처럼 원문 대응과 내용을 알 수 있는 이름을 사용한다.
6. **삽입**: 해당 Method·Experiments 문단 중간에 사이트 루트 경로 `/assets/img/...`로 참조한다. 필요한 경우 HTML 이미지 폭을 조절하고, 그림·표 바로 옆에서 무엇을 읽어야 하는지 설명한다.
7. **검수**: 저장한 파일을 직접 열어 crop 경계와 글씨·숫자 가독성을 확인하고 Markdown의 경로·파일명·panel 설명을 대조한다. 가능하면 실제 글의 미리보기에서 배치도 확인한다. 원본 이미지 확인만 했다면 웹 렌더링까지 검증했다고 보고하지 않는다.

해상도는 작은 글씨와 표 숫자를 읽을 수 있도록 잡는다. PDF 렌더링은 보통 200~300 DPI부터 확인하고 필요하면 높인다. 디스플레이 screenshot의 해상도만으로 원본 품질을 제한하지 않는다. 기존 asset 파일은 요청 범위 안에서만 수정하며 새 포스팅의 파일로 무심코 덮어쓰지 않는다.

### 기본 이미지

```markdown
![두 단계 학습 파이프라인](/assets/img/YYYY-MM-DD-주제/pipeline.png)

첫 단계는 representation을 학습하고 두 번째 단계는 이를 caption으로 해석한다.
```

그림은 설명할 문단 옆에 둔다. 순서는 그림을 먼저 보고 해석하거나, 실험 목적을 먼저 읽고 그림을 보는 두 형태 모두 가능하다. 그림만 여러 장 나열하고 해석을 생략하지 않는다. 여러 panel이면 `(a)`, `(b)`, 좌/우 또는 색상·범례를 지정해 설명한다.

`/assets/img/...`의 앞 슬래시는 사이트 루트 경로다. 실제 폴더명과 파일의 대소문자를 맞춘다. 공백이 있는 경로는 기존 글처럼 `%20`로 쓸 수 있다. alt text는 최근 글에 screenshot 이름이 많지만 새 그림은 기능을 알 수 있는 설명을 붙인다.

### HTML 이미지와 가로 비교

```html
<img src="/assets/img/YYYY-MM-DD-주제/result.png"
     alt="정렬 유무에 따른 실험 결과"
     style="width: 70%;" />
```

기존 글에는 `style="zoom: 25%;"`, `33%`, `50%`, `67%` 등도 사용된다. 같은 문맥의 기존 이미지를 보완할 때는 인접 형식과 맞춘다. 새 그림에서는 읽기 좋은 폭을 선택한다.

```html
<div style="display: flex; justify-content: space-between; align-items: center;">
  <img src="/assets/img/YYYY-MM-DD-주제/left.png"
       alt="첫 번째 비교 조건" style="width: 50%;">
  <img src="/assets/img/YYYY-MM-DD-주제/right.png"
       alt="두 번째 비교 조건" style="width: 50%;">
</div>
```

SD3에 실제 사용된 가로 비교 형식이다. 두 이미지의 글씨가 작아지면 폭을 강제하지 않고 세로로 배치한다. 서로 다른 그림에 같은 alt text를 복사하지 않는다.

### 논문 표 이미지와 Markdown 비교 표

원문 결과 표는 대부분 이미지로 넣고, 핵심 결과를 평문·bullet로 읽어준다. 작성자가 개념을 재구성한 비교는 Markdown table로 쓸 수 있다.

```markdown
| 항목 | 기존 방식 | 제안 방식 |
| --- | --- | --- |
| 예측 단위 | 단일 token | scale별 token map |
| 처리 순서 | token 단위 순차 | scale 단위 순차 |
```

표 앞뒤에는 빈 줄을 둔다. 긴 논리적 설명을 셀 안에 모두 몰지 않고 표 아래에서 해설한다. 셀 안에서 구분자로 오해할 수 있는 vertical bar는 escape하고, 수식의 조건부 표기는 `\mid`처럼 사용한다. 코드 block과 table은 실제 렌더링 방식이 다르므로 서로 대체하지 않는다.

## 링크와 코드

서지 링크는 바깥 괄호가 한 겹 더 있는 `[[표시명](URL)]`, 일반 참고 링크는 `[표시명](URL)`을 기본으로 한다. 기존 글 내부 링크는 해당 개념을 설명하는 문장 근처에 넣는다. 코드 링크는 관련 함수나 연산 바로 옆에 둔다.

inline code는 함수·변수·설정 이름에 사용한다. 여러 줄 코드는 언어를 지정한 fence에 넣고, 코드 밖에서 이유와 연산 순서를 설명한다.

````markdown
```python
t = sigmoid(u)
```

실수축의 표본을 0과 1 사이로 바꾸는 연산이다.
````

기존 SigLIP에는 언어를 지정하지 않은 fence가 있고, SD3에는 인용문 안의 `python` fence가 있다. 새 Python 코드는 언어를 지정한다. 의사코드를 실행 가능한 코드라고 소개하지 않는다. 예시의 코드·이미지·링크 placeholder는 실제 포스팅 완료 전에 모두 실제 내용으로 바꾼다.

## 기타 표기와 escape

- 초기 GAN·DDPM에는 `~~취소선~~`으로 개인적인 메모나 부연을 취소 표시한 사례가 있다. 새 글에서 작성자의 과거 생각이나 심경을 임의로 만들어 취소선으로 넣지는 않는다.
- 소스에는 `IBA\*`처럼 문자 그대로의 별표를 표시하려는 escape가 있다. literal 별표와 bold·italic marker를 구별한다. LaTeX의 `_`는 수식의 subscript이므로 Markdown escape를 기계적으로 복사하지 않는다.
- 코드·모델의 literal token인 `<image>`, `<eos>`는 backtick으로 감싸 실제 HTML 태그와 구별한다. 설명 문맥에서 HTML의 특수 문자를 문자 그대로 보여줄 때는 필요한 entity를 사용한다.
- SD3의 `&gt;`와 초기 글의 `&nbsp;` 사용도 확인했다. 일반 수학적 비교는 dollar 수식 안의 기호로, 문단·목록 배치는 실제 빈 줄·들여쓰기로 표현하는 것이 기본이다. 공백 entity를 반복해 목록 구조를 대체하지 않는다.
- literal 달러 화폐액과 수식 delimiter를 구별한다. 화폐 기호를 일반 문자로 보여줄 때는 필요에 따라 `\$`로 escape한다.
- 본문 조사에서 reference-style link 정의, autolink용 `<https://...>`, HTML heading, tilde fence, task list를 작성 관례로 발견하지 않았다. 이를 임의로 기존 블로그의 필수 문법이라고 추가하지 않는다. 이 내부 명세의 검수 체크박스는 포스팅 본문에 체크박스를 요구한다는 뜻이 아니다.

## 빈 줄과 줄바꿈

- H1/H2/H3와 본문 사이, 문단 사이, 이미지·표·수식·코드의 앞뒤에 빈 줄을 둔다.
- 한 인용문 내부에서 문단을 나누려면 빈 `>` 행을 둔다. 빈 일반 행으로 인용문을 끊는 것과 구별한다.
- 최근 글의 원본에는 여러 빈 줄이 연속되는 경우가 많다. 관찰한 소스 형식으로 기록하되 빈 줄 수 자체를 3개·5개로 고정하지 않는다. 논리적인 블록 분리를 반드시 유지한다.
- 문단 안에서 실제 줄바꿈이 필요할 때는 기존 글의 행 끝 공백 두 개 또는 `<br>`를 사용할 수 있다. 일반 설명은 빈 줄로 나눈 문단이 기본이다.
- `\\`는 LaTeX 수식 안의 줄바꿈에 사용한다. 일반 문단 줄바꿈용으로 쓰지 않는다.
- front matter의 `---`와 본문의 horizontal rule은 문자가 같아도 역할이 다르다. front matter를 결과적으로 본문 구분선처럼 취급하지 않는다.

## 조사한 사용 빈도와 과거 변형

집계는 front matter와 fenced code 내용을 제외한 본문 소스의 문법 검출이다. 숫자는 **해당 문법이 나타난 포스팅 수**이며 사용 횟수나 렌더링 성공 횟수가 아니다. 최신 표본은 앞선 원문 대조의 7편과 같다.

| 문법 | 전체 39편 | 최신 7편 | 해석 |
| --- | ---: | ---: | --- |
| `$$` display math | 32 | 7 | 현재 수식 표기의 공통 형식 |
| 독립 행 `\[` 또는 `\]` | 1 | 0 | SAGAN의 과거 예외; 새 글에서는 사용하지 않음 |
| `>` 인용문 | 38 | 7 | 서지와 본문 보충 모두에 사용 |
| 중첩 `> >` | 4 | 2 | 짧은 추가 정의·해설 |
| `**bold**` | 38 | 7 | 제목·주요 개념·라벨의 공통 강조 |
| `-` bullet | 23 | 7 | 기호·조건·결과 등의 병렬 정보 |
| `*` bullet | 34 | 7 | 역할·기여·결과 등에 함께 사용 |
| 숫자 목록 | 32 | 7 | 단계·알고리즘·기여 |
| Markdown 이미지 | 38 | 7 | 기본 figure 배치 |
| HTML `<img>` | 27 | 2 | 크기 조정 등의 변형 |
| HTML `<details>` | 15 | 2 | 긴 추가 설명을 접는 기존 형식 |
| `<br>` | 7 | 2 | 토글 뒤 등의 간격 |
| `[[표시명](URL)]` | 23 | 7 | 최근 서지 링크의 공통 형식 |
| fenced code block | 7 | 2 | 코드·의사코드가 필요한 경우 |
| inline code | 7 | 2 | 실제 코드 식별자·설정 |
| Markdown table | 4 | 0 | DINO·Flamingo·AnyGPT·WGAN의 비교 정리 |
| `\tag{...}` | 18 | 5 | 원문 수식 대응 |
| `align` 등의 수식 환경 | 10 | 1 | 여러 줄 유도 |
| 본문의 `---` | 6 | 1 | 일부 설명 묶음 구분 |
| 행 끝 두 공백 | 13 | 5 | 명시적 줄바꿈이 필요한 일부 문맥 |
| `<span>` 색상 강조 | 9 | 0 | 이전 글의 강조 변형 |
| `<u>` 밑줄 | 2 | 0 | 일부 과거 글의 강조 변형 |
| `~~취소선~~` | 2 | 0 | 초기 GAN·DDPM의 취소 표시 |
| HTML entity | 4 | 1 | `&gt;`, `&nbsp;` 등의 문자·공백 표현 |
| 이미지 attribute list `{: ...}` | 2 | 0 | Transformer·StarGAN의 과거 변형 |
| Gist `<script>` | 1 | 0 | DCGAN의 외부 코드 삽입 |

전체 조사에서는 `\(...\)` 인라인 delimiter, Markdown footnote, Liquid template, HTML comment를 본문 작성 방식으로 발견하지 않았다. 이 결과를 근거로 임의의 callout 문법, 주석 체계, 발주용 인용 체계를 블로그의 필수 스타일로 추가하지 않는다.

과거의 `<b>`, `<i>`, `<span>`, `<u>`, `<blockquote>`, `<ul>/<ol>/<li>`, `<figure>/<figcaption>`, `<aside>`, HTML `<a>`도 기록했다. 최신 기본은 Markdown bold·italic·quote·목록과 위의 제한된 HTML이다. 기존 글에서 이런 HTML을 보완할 때는 형식을 유지할 수 있지만, 새 글에 과거 변형을 모두 동시에 도입하지 않는다. literal token인 `<eos>`, `<image>` 등을 실제 HTML 작성 문법으로 잘못 집계하지 않는다.

## 실제 파일에서 확인할 근거

| 확인할 형식 | 원본 파일과 행 |
| --- | --- |
| H1과 서지정보·링크 | [Diffusion Bridge](../_posts/Paper%20Reivew/Multimodal/2026-02-19-DiffusionBridge.md) L12, L14–20 |
| H2 Background와 추가 H3 개념 | [Diffusion Bridge](../_posts/Paper%20Reivew/Multimodal/2026-02-19-DiffusionBridge.md) L26 이후 |
| 수식 묶음 뒤 기호 목록 | [Diffusion Bridge](../_posts/Paper%20Reivew/Multimodal/2026-02-19-DiffusionBridge.md) Eq. 1–5 아래 |
| 원문 구현 상세를 설명용 subsection으로 분리 | [Diffusion Bridge](../_posts/Paper%20Reivew/Multimodal/2026-02-19-DiffusionBridge.md)의 Decoder 입력 어댑터 소절 |
| 긴 유도와 비교의 HTML 토글 | [SD3](../_posts/Paper%20Reivew/Multimodal/2025-12-01-SD3.md)의 FM → CFM, MM-DiT vs Cross Attention |
| H5의 기술 요소와 중첩 인용 | [SD3](../_posts/Paper%20Reivew/Multimodal/2025-12-01-SD3.md) L868, L883 이후 |
| 두 이미지의 HTML 가로 배치 | [SD3](../_posts/Paper%20Reivew/Multimodal/2025-12-01-SD3.md) L747–751 |
| bold 실험 라벨과 결과 bullet | [VLV AE](../_posts/Paper%20Reivew/Multimodal/2026-01-14-VLVAE.md)의 Main Results |
| 별도 핵심 기호 목록과 코드 줄별 해설 | [SigLIP](../_posts/Paper%20Reivew/Multimodal/2025-09-02-SigLIP.md)의 Method |
| 토글 안의 비교 table | [DINO](../_posts/Paper%20Reivew/Computer%20Vision/2025-08-05-DINO.md) L1279 이후 |
| 추가 H3와 짧은 nested 설명 | [BERT](../_posts/Paper%20Reivew/NLP/2025-07-24-BERT.md)의 Introduction 목록 |
| 과거 bracket display 예외 | [SAGAN](../_posts/Paper%20Reivew/Computer%20Vision/GAN/2024-03-30-SAGAN.md) L93–97 |
| 과거 이미지 attribute와 aside | [Transformer](../_posts/Paper%20Reivew/NLP/2023-08-14-Transformer.md) L15, L140–144 |

이 표의 목적은 소스 문법의 근거를 보여주는 것이다. 이전 원문 대조 대상 밖의 논문 내용을 새로 검증했다는 뜻은 아니다.

## 형식 검수 목록

- [ ] 포스팅을 적절한 주제 디렉토리에 저장하고 필요한 새 디렉토리를 생성했다.
- [ ] front matter의 분야·tags와 실제 저장 깊이에 따른 typora-root-url이 맞다.
- [ ] 파일 첫머리의 front matter와 본문 H1이 서로 다른 역할로 작성됐다.
- [ ] 본문 H1은 하나이며 `# **[논문 리뷰] ...**` 형식이다.
- [ ] 서지정보는 `>` 안에서 제목·저자·학회·링크 순서로 기입했다.
- [ ] 주요 장은 H2이며 Background부터 유사한 설명 흐름을 유지한다.
- [ ] 이해를 위해 새로 나눈 소절은 H3이고 H4/H5의 실제 상하위 관계가 맞다.
- [ ] 논문의 모든 절과 부록을 제목으로 기계적으로 복사하지 않았다.
- [ ] 중요한 식은 독립 `$$` 블록, inline 식은 `$...$`이다.
- [ ] 중요한 기호는 식 또는 식 묶음 바로 아래의 `- $기호$: 설명`으로 정의했다.
- [ ] 긴 부연은 details로 접고 summary 제목·markdown div·닫는 태그를 확인했다.
- [ ] quote·목록 안의 수식·그림·코드는 prefix와 들여쓰기가 유지된다.
- [ ] 짧은 라벨·bold·italic·inline code·수학 기호의 역할이 구별된다.
- [ ] 순서 있는 과정과 병렬 목록을 구분하고 한 목록의 marker가 일관된다.
- [ ] 그림·표·코드 앞뒤의 빈 줄과 본문 해설을 남겼다.
- [ ] 필요한 figure·table을 실제로 캡처·추출해 assets에 저장하고 해당 설명에 삽입했다.
- [ ] 저장 이미지의 글씨·숫자·축·범례·crop 경계를 직접 확인하고 원문 번호·panel과 대조했다.
- [ ] 원문 표 이미지와 재구성한 Markdown 비교 표를 구별했다.
- [ ] 경로·링크·식 번호·escape·fence·HTML의 짝이 맞는다.
- [ ] 과거 색상·밑줄·Gist 등의 변형을 새 글의 기본 스타일로 무작정 추가하지 않았다.
- [ ] `bundle exec jekyll serve`로 생성한 실제 글을 브라우저에서 열고 닫힌 토글 내부까지 확인했다.
- [ ] 최종 수정본의 수식·asset·표·코드·목록·레이아웃이 정상이고 Markdown·로컬 미리보기를 사용자에게 보여줄 준비가 됐다.

소스 문법 확인 후에는 [작성 기준의 로컬 렌더링 검증](paper-review-style.md#로컬-렌더링-검증과-최종-확인)을 반드시 수행한다. 사용자 최종 컨펌과 명시적 GitHub 푸시 승인 전에는 로컬 결과를 검토할 수 있는 상태로 제공한다.
