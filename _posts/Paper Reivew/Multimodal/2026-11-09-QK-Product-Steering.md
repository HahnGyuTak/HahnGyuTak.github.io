---
title: "[Paper Review] QK Product Steering"
description: "VLM의 query와 key가 만드는 attention 연산자에서 상위 singular mode를 줄여 객체 환각을 완화하는 방법과, 그 효과가 성립하는 조건을 정리한다."
date: 2026-11-09 17:11:00 +0900
categories: [Paper Review, Multi Modal, EMNLP2026]
tags: [Multi-modal, VLM, Hallucination, Attention, Weight Editing, Mechanistic Interpretability]
use_math: true
typora-root-url: ../../../
---

# **[논문 리뷰] Spectral Query-Key Product Weight Steering for Training-Free VLM Hallucination Mitigation**

> **Spectral Query-Key Product Weight Steering for Training-Free VLM Hallucination Mitigation**
>
> Karn Tiwari, Varnith Chordia, Prathosh A P
>
> Findings of EMNLP 2026 · COLM 2026 Workshop on Actionable Interpretability
>
> [[arXiv](https://arxiv.org/abs/2606.20419v2)] [[Paper](https://aclanthology.org/2026.findings-emnlp.742/)]

## **Background**

### **Object Hallucination: 그럴듯한 설명과 실제로 있는 객체**

이미지에 샌드위치와 샐러드가 있으면 식탁에 포크가 있을 법하다. 모델이 실제 이미지에 포크가 있는지 확인하지 않고 포크를 설명에 넣으면 **object hallucination**이다. 문장이 자연스럽다는 것과 이미지가 그 문장을 뒷받침한다는 것은 별개의 문제다.

객체를 잘못 알아봐서 이런 오류가 생길 수도 있고, 학습 중 많이 본 객체 조합을 시각 증거보다 강하게 따를 수도 있다. 이 논문은 후자의 가능성에 주목한다. **이미 갖고 있는 시각 정보를 더 잘 쓰도록 기존 attention을 바꿀 수 있는지** 묻는다.

### **Attention에서 Query와 Key가 함께 결정하는 것**

한 layer에서 query head $h$가 key head $g$를 참조한다고 하자. 토큰을 행으로 쌓은 hidden state를 두 projection에 통과시키면 다음과 같다.

$$
Q_h=XW_{q,h}^{\top},\qquad K_g=XW_{k,g}^{\top}
\tag{1}
$$

- $X\in\mathbb{R}^{n\times d}$: $n$개 토큰의 hidden state. 각 토큰은 $d$차원이다.
- $W_{q,h},W_{k,g}\in\mathbb{R}^{r\times d}$: 해당 query와 key의 projection weight. $r$은 head dimension이다.
- $Q_h,K_g\in\mathbb{R}^{n\times r}$: 입력에 따라 달라지는 query와 key activation

이 둘을 내적하면 $W_{q,h}$와 $W_{k,g}$가 따로 나타나지 않고, 아래처럼 곱으로 묶인다.

$$
Q_hK_g^\top
=XW_{q,h}^{\top}W_{k,g}X^\top
=XM_hX^\top
\tag{2}
$$

- $M_h=W_{q,h}^{\top}W_{k,g}\in\mathbb{R}^{d\times d}$: 두 token의 내용이 내적 점수로 바뀌는 방식을 정하는 **QK product**. 원문 Eq. (3)의 정의다.
- $Q_hK_g^\top\in\mathbb{R}^{n\times n}$: query 위치와 key 위치 쌍에 대한 내적 점수

예를 들어 토큰 $i$가 $j$를 읽는 점수의 내용 항은 $x_i^\top M_h x_j$다. $M_h$는 prompt를 넣고 나서 얻은 attention map이 아니라, **weight만으로 계산할 수 있는 연산자**다. 두 행렬 사이의 head dimension이 $r$이므로 $M_h$의 rank는 최대 $r$이다. 겉보기에는 $d\times d$지만 큰 행렬의 모든 방향이 독립적으로 쓰이는 것은 아니다.

<details><summary>실제 attention logit에서도 M 하나로 모든 위치를 설명할 수 있는가?</summary>
  <div markdown="1">

Eq. (2)는 RoPE를 적용하기 전의 내용 항을 전개한 것이다. 위치 $i,j$에서 query와 key에 각각 회전을 적용하면 두 회전의 상대적인 차이도 내적에 들어간다.

$$
\ell_{ij}^{\mathrm{RoPE}}
=\frac{x_i^\top W_{q,h}^\top R_i^\top R_j W_{k,g}x_j}{\sqrt r}
$$

- $x_i,x_j\in\mathbb{R}^{d}$: 각 토큰의 hidden state를 열벡터로 쓴 것
- $R_i,R_j\in\mathbb{R}^{r\times r}$: 해당 위치의 RoPE 회전. 1D 회전으로 단순화한 표기다.
- $\ell_{ij}^{\mathrm{RoPE}}$: 회전과 head scaling을 적용한 점수. causal mask와 softmax는 이후 적용한다.

따라서 이 방법이 편집하는 것은 **위치에 따라 달라지는 전체 logit이 아니라 projection weight에 들어 있는 내용 상호작용**이다. 모델을 실행할 때는 원래 RoPE와 mask가 계속 적용된다.

  </div>
</details>

<br>

### **Grouped-Query Attention과 Shared Key**

Grouped-query attention(GQA)에서는 여러 query head가 하나의 key/value head를 공유한다. 예를 들어 Qwen2.5-VL-7B의 query head 28개는 key/value head 4개를 공유하므로 그룹당 query head가 7개다.

이 구조에서 하나의 query head만 다루고 싶어도, 그 head가 사용하는 **공유 key weight를 바꾸면 같은 그룹의 다른 query도 함께 영향을 받는다.** 따라서 논문의 편집은 $M_h$를 목표로 삼되 실제 checkpoint에서는 그 head의 $W_{q,h}$만 바꾼다. 다른 head의 key를 건드리지 않는다는 뜻이며, 뒤쪽 layer나 최종 답변에 효과가 전달되지 않는다는 뜻은 아니다.

## **Introduction**

VLM의 객체 환각을 줄이는 접근은 개입 시점이 다르다. 추가 학습은 데이터와 최적화가 필요하고, VCD 같은 decoding 개입은 생성하는 동안 logits를 조정한다. 이 논문은 **기존 checkpoint의 일부 query weight를 한 번 수정한 뒤, 같은 모델 구조와 decoding 절차로 실행**한다.

### **Goal**

Caption에 없는 객체를 덧붙이는 빈도를 낮추면서, 추론 중 추가 forward나 검증 단계를 만들지 않는다. 또한 QK product의 어느 방향을 바꿨을 때 효과가 나는지 진단한다.

### **Motivation**

논문의 가설은 큰 singular mode가 강한 vision-language prior를 담을 수 있다는 것이다. 예를 들어 특정 이미지 내용과 앞서 생성된 단어들이 높은 점수로 연결되면, 문맥상 그럴듯한 객체가 실제 증거보다 쉽게 선택될 수 있다.

큰 singular value가 곧 환각 원인이라는 뜻은 아니다. 논문은 **상위 모드와 같은 개수의 무작위·하위 모드를 편집했을 때 결과가 달라지는지** 비교해 가설을 평가한다.

[VLM Localization 논문](https://arxiv.org/abs/2605.19792v1)의 head ablation은 어떤 연산에 모델이 의존하는지 알아보기 위해 실행 중에 head 출력을 끈다. 여기서는 head의 일부 singular mode를 약화한 뒤 **수정한 weight를 실제로 저장**한다. 어느 경로를 쓰는지 진단하는 실험과, 그 경로를 편집해 오류를 줄이는 방법의 차이다.

### **Contributions**

1. Head별 QK product의 상위 singular mode를 골라 줄이는 **QK Product Steering**을 제안한다.
2. 원하는 QK 변화를 **query-only update**로 옮겨 GQA의 shared key를 유지한다.
3. QK를 대칭·반대칭 성분으로 나누고, 편집 후 attention과 benchmark가 어떻게 바뀌는지 비교한다.

## **Method**

### **전체 흐름: Operator를 편집하고 Query Weight로 되돌린다**

![Figure 1: QK product 구성, 상위 singular mode 억제와 query weight 복원](/assets/img/2026-11-09-QK-Product-Steering/fig-01-overview.png)

Figure 1의 흐름은 **현재 query/key weight → QK product의 spectrum → 목표 QK product → 새 query weight**다. 논문의 기본 설정은 중간 3분의 1 layer에 있는 모든 query head에 적용한다. 여기서 top-$k$는 head나 token의 개수가 아니라, **각 head에서 편집할 singular mode 개수**다.

### **1. 작은 행렬에서 Thin SVD 구하기**

$d\times d$인 $M_h$에 큰 SVD를 직접 적용할 필요는 없다. $W_{q,h}^\top$과 $W_{k,g}^\top$에 economy QR을 적용하고, 그 안쪽의 작은 행렬을 분해한다.

$$
\begin{aligned}
W_{q,h}^{\top}&=Q_qR_q,\qquad W_{k,g}^{\top}=Q_kR_k,\\
C_h&=R_qR_k^\top=\widehat U\Sigma\widehat V^\top,\\
M_h&=Q_qC_hQ_k^\top=U\Sigma V^\top,\\
U&=Q_q\widehat U,\qquad V=Q_k\widehat V.
\end{aligned}
$$

- $Q_q,Q_k\in\mathbb{R}^{d\times r}$: QR에서 얻은 직교 열 기저. 앞 절의 query activation $Q_h$와 구분한다.
- $R_q,R_k\in\mathbb{R}^{r\times r}$: QR의 작은 삼각 행렬
- $C_h\in\mathbb{R}^{r\times r}$: 실제로 SVD를 계산할 행렬
- $\widehat U,\widehat V$: $C_h$의 좌·우 singular vector 행렬
- $U,V\in\mathbb{R}^{d\times r}$: 작은 SVD의 방향을 원래 hidden space로 옮긴 좌·우 기저
- $\Sigma=\operatorname{diag}(\sigma_1,\ldots,\sigma_r)$: 내림차순 singular value. 실제 rank가 더 작으면 0도 포함된다.

$Q_q,Q_k$는 직교 열을 가지므로 $C_h$의 singular value가 보존된다. 원문 Eq. (12)–(14)는 이 방법으로 **큰 hidden dimension 대신 head dimension 크기의 SVD**를 계산한다.

### **2. 상위 Mode를 약화하기**

$M_h$를 구성하는 각 모드는 좌·우 방향 $u_i,v_i$와 크기 $\sigma_i$를 갖는다. 가장 큰 $k$개만 골라 그 계수에 $1-\alpha$를 곱한다.

$$
\begin{aligned}
\Sigma_{ii}^\star&=
\begin{cases}
(1-\alpha)\sigma_i, & i\le k,\\
\sigma_i, & i>k,
\end{cases}\\
M_h^\star&=U\Sigma^\star V^\top,\\
\Delta M_h&=-\alpha\sum_{i=1}^k\sigma_i u_i v_i^\top.
\end{aligned}
$$

- $k$: 편집할 상위 모드 개수
- $\alpha$: 모드 억제 강도
- $u_i,v_i$: $U,V$의 $i$번째 열
- $M_h^\star$: 원하는 목표 product
- $\Delta M_h=M_h^\star-M_h$: 목표 변화량

$\alpha=0$이면 원본 그대로이고, $0<\alpha<1$이면 일부만 줄이며, $\alpha=1$이면 고른 모드의 계수를 0으로 만든다. $\alpha>1$에서는 계수가 음수가 되어 해당 모드의 **부호를 뒤집는다.** 그래서 강한 편집의 결과를 단순히 더 많이 제거했다고 설명하면 안 된다.

같은 $k$라도 layer마다 잘라내는 정보량은 다르다. 상위 모드에 모인 비중을 다음처럼 계산한다.

$$
E_k(M_h)=
\frac{\sum_{i=1}^k\sigma_i^2}{\lVert M_h\rVert_F^2},
\qquad
\frac{\lVert\Delta M_h\rVert_F}{\lVert M_h\rVert_F}
=|\alpha|\sqrt{E_k(M_h)}
$$

- $E_k(M_h)$: 상위 $k$개가 차지하는 **Frobenius norm 제곱의 비율**
- $\lVert\cdot\rVert_F$: Frobenius norm

예를 들어 $E_3=0.10$이고 $\alpha=1$이면 제곱 에너지의 10%를 제거한다. 이를 attention 확률이나 연산량이 10% 줄었다는 뜻으로 읽으면 안 된다. 이 계산은 데이터 없이도 **같은 top-3 편집이 구조적으로 어느 layer에서 상대적으로 작은 변화인지** 비교하게 해 준다.

<details><summary>변화량이 작은 편집이면 최종 답변도 유지되는가?</summary>
  <div markdown="1">

부록 B.6에서 입력 $X$를 고정한 한 head의 내용 logit 변화는 다음과 같이 제한된다.

$$
\lVert X\Delta M_h X^\top\rVert_F
\le \lVert X\rVert_2^2\,\lVert\Delta M_h\rVert_F
$$

- $\lVert X\rVert_2$: 입력 행렬의 spectral norm
- $X\Delta M_hX^\top$: 목표 product를 적용했을 때, 같은 $X$에서의 내용 logit 변화

작은 weight 변화가 특정 입력의 logit에 주는 영향을 제한하는 식이다. 최종 출력 보존을 말하려면 그 변화가 뒤쪽 연산에서 얼마나 커지는지와 **원래 정답이 다음 후보보다 얼마나 앞섰는지**까지 알아야 한다. 부록 B.7의 margin 조건도 이 두 요소에 의존한다. 식 하나로 일반 능력 보존을 보장할 수 없어 benchmark를 따로 본다.

  </div>
</details>

<br>

### **3. GQA에서 Key를 고정하고 Query Weight 복원하기**

목표 $M_h^\star$를 얻었지만 실제 모델은 $M_h$라는 독립 weight를 저장하지 않는다. 따라서 $W_{k,g}$를 고정하고, 새 query weight가 만드는 변화가 $\Delta M_h$에 가까워지도록 한다.

$$
\min_Z\quad
\lVert W_{k,g}^{\top}Z-\Delta M_h^\top\rVert_F^2
+\lambda\lVert Z\rVert_F^2
$$

- $Z\in\mathbb{R}^{r\times d}$: 구하려는 query weight 변화량 $\Delta W_{q,h}$
- $W_{k,g}^{\top}Z$: 실제 product 변화의 전치
- $\lambda$: 큰 query 변화와 수치 불안정을 줄이는 ridge 계수

이는 원문이 구현에 사용한 ridge 식을 **최소제곱 목적**으로 풀어 쓴 것이다. 첫 항은 목표 product를 따라가게 하고, 두 번째 항은 그 과정에서 weight가 필요 이상으로 변하는 것을 억제한다. 미분을 0으로 두면 $r\times r$ 선형계가 나오고, 해는 다음과 같다.

$$
\Delta W_{q,h}=
\bigl(W_{k,g}W_{k,g}^{\top}+\lambda I\bigr)^{-1}
W_{k,g}\Delta M_h^{\top}
\tag{20}
$$

- $\Delta W_{q,h}$: 해당 head의 $W_{q,h}$에 더할 update
- $I\in\mathbb{R}^{r\times r}$: 항등행렬
- $\lambda=\epsilon\,\operatorname{tr}(W_{k,g}W_{k,g}^{\top})/r$: Eq. (21)의 scale. $\epsilon$은 ridge 강도를 조절한다.

마지막으로 $W_{q,h}\leftarrow W_{q,h}+\Delta W_{q,h}$를 저장한다. Key와 value weight, 모델의 크기, token 수, KV cache 구조는 그대로다. 논문은 A100 80GB에서 이 일회성 편집에 약 **6–10초**가 들었다고 보고한다. 여기서 **zero inference cost**는 편집한 모델이 추론할 때 추가 연산 경로를 넣지 않는다는 의미다.

<details><summary>Query만 바꿔도 목표 QK product를 정확히 만들 수 있는가?</summary>
  <div markdown="1">

원문 Eq. (19)의 등식 제약과 실제 Eq. (20)의 ridge 해를 구분해야 한다. Key를 고정하면 새로 만들 수 있는 변화는 $\Delta W_{q,h}^{\top}W_{k,g}$ 형태뿐이다. 따라서 그 행들은 언제나 $W_{k,g}$의 row space에 놓인다.

전체 $M_h$의 SVD에서 양의 singular value를 갖는 우 singular vector는 이 row space 안에 있다. 따라서 해당 모드를 직접 줄인 $\Delta M_h$는 이 조건을 만족한다. Key가 full row rank이고 $\lambda=0$이면 목표 product를 정확히 실현할 수 있다.

실제 ridge에서는 아래처럼 오른쪽에 shrinkage가 들어간다.

$$
\Delta M_{\mathrm{real}}
=\Delta M_h\,W_{k,g}^{\top}
\bigl(W_{k,g}W_{k,g}^{\top}+\lambda I\bigr)^{-1}
W_{k,g}
$$

- $\Delta M_{\mathrm{real}}=\Delta W_{q,h}^{\top}W_{k,g}$: checkpoint에 update를 적용해 실현한 product 변화
- $W_{k,g}^\top(W_{k,g}W_{k,g}^\top+\lambda I)^{-1}W_{k,g}$: $\lambda=0$이고 full row rank이면 key row space의 직교 projection이다. $\lambda>0$이면 그 공간 안에서도 방향에 따라 축소된다.

뒤에서 다룰 대칭·반대칭 성분을 따로 수정하면 목표 변화가 이 공간 밖에도 놓일 수 있다. 그때는 **가능한 query-only 근사**다. 목표를 대칭으로 정했다고 해서 실제 변화까지 완전히 대칭이라고 보장할 수는 없다.

  </div>
</details>

<br>

### **4. Symmetric과 Antisymmetric 성분으로 비교하기**

전체 $M_h$의 top mode를 줄이면 대칭·반대칭 양쪽이 함께 바뀔 수 있다. 논문은 어떤 종류의 내용 점수에 효과가 더 가까운지 보기 위해 $M_h$를 둘로 나눈다.

$$
M_h=S_h+\Omega_h,\qquad
S_h=\frac{M_h+M_h^\top}{2},\quad
\Omega_h=\frac{M_h-M_h^\top}{2}.
$$

- $S_h$: 대칭 성분. $S_h^\top=S_h$이다.
- $\Omega_h$: 반대칭 성분. $\Omega_h^\top=-\Omega_h$이다.

두 token의 순서를 바꿨을 때 $x_i^\top S_hx_j$는 같고, $x_i^\top\Omega_hx_j$는 부호가 바뀐다. 즉, 대칭 부분은 **교환해도 같은 내용 점수**, 반대칭 부분은 **어느 쪽이 query인지에 따라 달라지는 내용 점수**를 나타낸다. Causal mask와 각 query 행의 softmax까지 적용한 attention 확률이 서로 같다는 뜻은 아니다.

| 진단 방식 | 목표로 바꾸는 성분 |
| --- | --- |
| QK Product Steering | 전체 $M_h$의 상위 singular mode |
| Sym-only | $S_h$의 상위 eigenmode |
| Antisym-only | $\Omega_h$의 상위 singular mode |
| Sym+Antisym | 두 성분을 각각 편집 |

원문의 대칭·반대칭 성분은 Frobenius 내적에서 직교하므로 편집 전 제곱 에너지는 합으로 나뉜다. 이 성질과 실제 모델에서 두 효과가 독립적이라는 주장은 다르다. 이후 비교는 동일한 query-only 복원 절차를 사용한 **편집 변형 간 비교**로 읽어야 한다.

## **Experiments**

### **Experimental Setup과 지표**

Qwen2.5-VL-7B, InternVL3-8B, Pixtral-12B를 대상으로 평가한다. Qwen에서 주로 sweep을 수행하고, 기본 설정은 중간 3분의 1과 $\alpha=1$을 유지하되 $k$를 Qwen 3, InternVL 1, Pixtral 3으로 선택한다. **편집 계산에는 이미지가 필요 없지만, 최종 설정 선택과 성능 평가는 CHAIR 데이터를 사용했다.**

주 실험은 COCO 2014 validation의 4,946장에 대해 greedy caption을 생성한다. [CHAIR](https://aclanthology.org/D18-1437/)는 생성된 객체 언급을 COCO 참조 정보와 대조한다.

- **CHAIR$_s$ ↓:** 환각 객체가 하나 이상 포함된 caption의 비율
- **CHAIR$_i$ ↓:** 인식한 전체 객체 언급 중 환각 언급의 비율
- **POPE / HallusionBench ↑:** 정해진 질문으로 객체 존재와 시각적 판단을 확인하는 probe
- **MME / MMMU ↑:** 일반 시각 이해·추론 능력의 변화 확인

없는 객체를 덜 말하는 것과, **있는 객체를 여전히 말하는 것**은 다르다. 예를 들어 모델이 객체를 하나도 말하지 않으면 CHAIR가 낮게 나타날 수 있다. 따라서 아래 layer 실험에서는 caption당 검출한 객체 수도 함께 본다.

### **1. 중간 Layer를 고르는 이유: 같은 Top-3의 영향이 다르다**

![Figure 2: 세 모델에서 레이어별로 상위 세 모드가 차지하는 QK 에너지 비율](/assets/img/2026-11-09-QK-Product-Steering/fig-02-spectral-energy.png)

Figure 2에서 선은 각 layer의 query head 평균, 색 영역은 head별 최소·최대, 회색 띠는 편집하는 중간 구간이다. 세 모델 모두 앞쪽의 일부 layer에서 $E_3$가 크게 올라가고, 중간에서는 낮아진다. Pixtral의 초반 최대치는 약 0.71이다. **모드 수가 똑같아도 초반에는 연산자의 더 큰 부분을 바꾸는 셈**이다.

![Table 3: Qwen에서 편집 위치에 따른 CHAIR와 캡션당 객체 수](/assets/img/2026-11-09-QK-Product-Steering/table-03-layer-ablation.png)

Qwen의 2,000장 비교에서 $k=3,\alpha=1$을 고정했다. Early 또는 전체 layer 편집은 $\Delta$CHAIR$_s=-28.81$pp로 겉보기에는 크게 개선되지만, caption당 COCO 객체 수가 **0.00**으로 무너진다. Middle은 baseline 2.54개에서 **2.67개**로 객체 언급을 유지하면서 $\Delta$CHAIR$_s=-3.04$pp였다. Late는 $+0.09$pp로 거의 움직이지 않았다.

스펙트럼으로 중간 layer를 후보로 정할 수 있지만, **낮은 에너지 비율만으로 정상적인 caption이나 낮은 환각을 보장할 수는 없다.** 이 비교가 왜 별도로 필요한지 보여주는 사례다.

### **2. Main Results: 개선 폭은 모델마다 다르다**

![Table 2: 세 VLM의 baseline, 무작위 모드, 상위 모드와 성분별 편집 결과](/assets/img/2026-11-09-QK-Product-Steering/table-02-main-results.png)

Table 2의 기본 full-product edit를 각 모델의 baseline과 비교하면 다음과 같다.

| 모델 | $k$ | CHAIR$_s$ 전 → 후 | 차이 |
| --- | ---: | ---: | ---: |
| Qwen2.5-VL-7B | 3 | 31.44% → 28.56% | −2.88pp |
| InternVL3-8B | 1 | 33.81% → 33.23% | −0.58pp |
| Pixtral-12B | 3 | 34.14% → 33.81% | −0.33pp |

모델별 상대 감소율은 약 **9.16%, 1.72%, 0.97%**다. 세 값의 평균은 **약 4.0%**이며, ACL Anthology의 Findings 초록에는 평균 상대 감소율이 **5.8%**로 적혀 있다. 표의 절대 감소 폭은 모델별로 0.33–2.88pp다.

같은 수의 무작위 mode를 줄인 대조군은 각각 31.33%, 33.97%, 34.00%로 변화가 작다. 다만 이 대조군은 **mode 개수**를 맞췄다. 상위 mode가 더 크므로, 이 비교만으로 편집 방향의 효과와 변화량 크기의 효과를 완전히 분리하기는 어렵다.

Sym-only는 Qwen에서 31.44% → 29.53%, InternVL에서 33.81% → 33.43%로 full edit 감소 폭의 약 3분의 2를 재현한다. Pixtral에서는 34.14% → 34.10%로 차이가 작다. Antisym-only는 Table 2의 설정에서 세 모델 모두 CHAIR$_s$가 오히려 올라간다. 이 결과는 **해당 설정의 효과가 대칭 성분에 더 가깝다**는 증거이며, 모든 모델에서 대칭 부분만으로 효과를 설명하거나 반대칭 부분을 불필요하다고 볼 근거는 아니다.

일반 능력 점수도 한 방향으로 움직이지 않는다. 예를 들어 Pixtral의 full edit는 MME가 2009.1 → 2018.6이지만 MMMU는 **45.44% → 43.11%**다. InternVL의 CHAIR$_i$는 62.98%로 같고, Qwen은 65.26% → 63.75%다. CHAIR$_s$와 일반 능력, 객체 언급당 지표를 따로 읽어야 한다.

<details><summary>어떤 개선이 신뢰구간에서도 남는가?</summary>
  <div markdown="1">

![Table 5: Qwen과 Pixtral에서 이미지 ID를 짝지어 추정한 CHAIR 변화의 95% 신뢰구간](/assets/img/2026-11-09-QK-Product-Steering/table-05-bootstrap.png)

같은 이미지에서 baseline과 편집 결과를 짝지어 bootstrap으로 변화량의 95% CI를 계산한다. 아래는 **별도 부록 Table 5의 조건**이며 Table 2의 점추정치와 섞지 않는다.

- **Qwen Sym-only, 1,000장:** $\Delta$CHAIR$_s=-2.17$pp, CI $[-2.95,-1.33]$. 반면 CHAIR$_i$ CI는 $[-1.44,+0.03]$으로 0을 포함한다. 부록 본문은 둘 다 0을 제외한다고 서술하지만, 이 글은 표의 구간을 따른다.
- **Pixtral full edit, 4,946장:** CHAIR$_s$ CI는 $[-0.85,+0.17]$로 0을 포함한다. CHAIR$_i$ CI는 $[-1.26,-0.33]$으로 0을 제외한다.
- Qwen의 해당 bottom-mode control과 Pixtral의 random control은 CI에 0이 포함된다. InternVL에 대해서는 이 표에 별도 CI가 없다.

따라서 세 모델의 CHAIR$_s$ 점추정치가 모두 내려간다는 것과, **세 모델 모두에서 CHAIR$_s$ 감소가 통계적으로 확인됐다는 것**은 구별한다.

  </div>
</details>

<br>

### **3. 얼마나 세게 편집할 것인가?**

Qwen의 1,000장 Sym top-3 실험에서 $\alpha$를 0.3, 0.5, 0.7, 1.0으로 올렸을 때 $\Delta$CHAIR$_s$는 각각 −0.56, −1.02, −1.39, −2.17pp였다. 표의 CI는 0.5부터 0을 제외한다. 반면 $k$만 바꾼 부록 표에서는 Sym top-5의 점추정치가 top-3보다 더 낮다(−2.93 vs. −2.17pp). Top-5에는 CI가 없어, **top-3가 점수 자체의 최적값이라고 말하기보다는 강도를 낮게 유지한 기본 설정**으로 이해하는 편이 정확하다.

![Table 9: Qwen에서 모드 수와 강도를 높였을 때의 일반 능력 변화](/assets/img/2026-11-09-QK-Product-Steering/table-capability-tradeoff.png)

500장 joint sweep에서 full-product edit의 가장 낮은 CHAIR$_s$는 $(k,\alpha)=(6,2)$의 **24.78%**다. 하지만 높은 강도는 다른 능력을 해칠 수 있다. 별도의 Table 9에서 기본 $(3,1)$의 MMMU는 50.89%이고, $k$만 6으로 늘리면 47.11%, $\alpha$만 2로 올리면 43.78%다. MME에서도 강도를 높인 쪽의 감소가 더 크다. **CHAIR 하나만 최적화한 설정과 일반 능력까지 고려한 설정이 다르다.**

### **4. Mechanism: 편집된 Layer보다 뒤에서 드러난 변화**

편집한 모델이 다른 caption을 생성하면 이후 입력 토큰도 달라진다. 토큰이 달라진 탓인지 weight를 바꾼 탓인지 분리하기 위해, 저자들은 **같은 prompt와 baseline caption을 두 모델에 teacher forcing**한다. 이후 attention에서 이미지 token들에 할당된 확률을 합해 edit − baseline 차이를 본다. 이 합이 커졌다는 것과 이미지 안의 어느 객체를 더 잘 봤다는 것은 구별한다.

![Figure 3a: 같은 caption prefix에서 측정한 Qwen 전체 레이어의 이미지 attention 변화](/assets/img/2026-11-09-QK-Product-Steering/fig-03a-attention-profile.png)

- **$L_0$–$L_8$:** 편집 전 layer라 변화가 0이다.
- **$L_9$–$L_{17}$:** 실제 편집 구간이다. Head마다 변화 방향이 달라 layer 평균은 거의 0이다.
- **$L_{18}$–$L_{27}$:** weight를 수정하지 않은 뒤쪽 layer다. Figure 3a에서는 $L_{18}$이 음수지만 이후는 대체로 양수이며, 구간 평균은 약 **+0.4pp**다.

Figure 3a의 캡션은 표본 수를 200으로, 그래프 제목은 20으로 표기한다. 위 해석은 그래프의 축과 구간 평균을 따른다.

![Figure 3b: 뒤쪽 레이어의 full-product, Sym-only, Antisym-only 비교](/assets/img/2026-11-09-QK-Product-Steering/fig-03b-downstream-attention.png)

Figure 3b에서 QK Product Steering과 Sym-only는 뒤쪽 구간에서 대체로 양의 이동을 보인다. Antisym-only는 일부 layer에서 증가하지만 뒤쪽까지 같은 방향의 변화가 이어지지는 않는다. 편집은 앞에서 일어나지만, 뒤쪽 layer는 **이미 다른 hidden state를 입력받기 때문에** weight가 같아도 attention이 달라질 수 있다.

<details><summary>편집 구간의 평균이 거의 0이면 실제로 아무 변화가 없는가?</summary>
  <div markdown="1">

![Figure 5: 편집 구간의 레이어와 헤드별 이미지 attention 변화](/assets/img/2026-11-09-QK-Product-Steering/fig-head-attention.png)

부록 Figure 5는 9개 layer의 28개 query head를 따로 보여준다. 편집 구간에서도 빨간색과 파란색이 섞여 있다. 양·음의 head 수는 캡션과 그래프의 표기가 다르므로, 여기서는 변화 방향이 섞여 있다는 점만 읽는다. **이미지에 준 attention 질량의 평균이 상쇄됐다고 해서 각 head가 읽은 value나 residual stream의 벡터 변화까지 상쇄되는 것은 아니다.**

이 결과는 편집이 attention을 바꿨다는 통제된 비교다. 반면 뒤쪽 이미지 attention 증가만 따로 막아 CHAIR 개선이 사라지는지 시험한 것은 아니므로, 그 증가가 환각 감소의 필수 매개라는 결론까지는 갈 수 없다.

  </div>
</details>

<br>

다른 가능성도 확인했다. 100장에 대한 고정 prefix 실험에서 환각 명사 토큰의 log-prob 변화는 −0.004, 근거 있는 명사 토큰은 −0.049다. 명사 직전의 이미지 attention도 대조군과 차이가 작다. 즉, **잘못된 명사만 골라 누르거나 그 직전에만 이미지를 더 읽는 패턴은 관찰되지 않았다.**

저자들은 뒤쪽 layer의 작은 이동이 긴 caption의 생성 경로를 따라 누적될 수 있다고 해석한다. 이는 teacher forcing과 자유 생성 결과를 잇는 가설이다. 짧은 답에서는 절대 영향을 주지 않는다거나, attention 증가가 곧 올바른 객체를 봤다는 뜻은 아니다.

### **5. Qualitative Example: 없던 객체를 빼면서 기존 내용을 남기는가?**

<img src="/assets/img/2026-11-09-QK-Product-Steering/fig-qualitative-example.jpg" alt="COCO 120853: 샌드위치와 코울슬로가 있는 야외 식탁" width="500">

부록 Example 1에서 baseline은 코울슬로 그릇에 흰 플라스틱 포크가 있다고 묘사한다. 논문은 이 포크 언급을 환각으로 분류한다. 편집 모델은 포크를 언급하지 않으면서 샌드위치, 그릇, 당근, 휴대폰, 식탁을 계속 묘사한다. Caption의 다른 표현도 함께 바뀌었으므로, 생성 도중 포크 단어 하나를 지웠다는 뜻은 아니다.

부록은 baseline에 있던 환각 객체를 하나 이상 빼고, 새로운 환각 객체를 넣지 않으며, 근거 있는 객체 언급은 유지한 사례를 4,946장 중 **756장** 찾았다고 보고한다. 이 이미지는 부록에 제시된 사례 중 하나다. 이는 **개선 기준에 맞춰 고른 예시**이므로 전체 데이터에서 모든 caption이 개선됐거나, 색·행동 등 COCO 객체 사전 밖의 내용까지 맞았다는 의미는 아니다.

## **Conclusion**

QK Product Steering은 개별 query나 key를 따로 볼 때 놓치기 쉬운 **두 projection의 곱**을 편집 대상으로 삼는다. 얇은 SVD로 각 head의 강한 방향을 찾고, 중간 layer에서 소수의 모드를 줄인 뒤, 공유 key를 유지하면서 query weight에 반영한다. 학습이나 생성 중 별도 처리를 추가하지 않고 caption 환각을 낮출 수 있다는 것이 핵심이다.

수치의 개선 폭은 Qwen에서 가장 크고 다른 두 모델에서는 작았다. 특히 **객체를 말하지 않아 CHAIR만 낮아진 경우**와 **객체를 계속 말하면서 틀린 언급을 줄인 경우**를 나눠 봐야 한다. 대칭 성분 편집과 뒤쪽 이미지 attention 변화는 작동 방식을 이해할 단서지만, 모든 VLM에 같은 회로가 있거나 해당 attention 변화가 반드시 필요하다는 증거는 아니다.

### **Limitations**

- **개입 범위:** 내용 기반 QK를 편집한다. 위치별 RoPE 상호작용을 직접 모델링하거나, encoder가 포착하지 못한 시각 정보를 새로 추가하지는 않는다.
- **모델·평가 범위:** 세 GQA VLM과 COCO 객체 환각이 중심이다. 관계·수량·텍스트 인식 오류나 다른 생성 설정에도 같은 효과가 있는지는 별도로 검증해야 한다.
- **강도와 능력의 균형:** 큰 $k$나 $\alpha$는 CHAIR를 더 낮출 수 있지만 일반 능력이 나빠질 수 있다. 모델별 $k$ 선택과 표본의 신뢰구간을 함께 봐야 한다.
- **편집의 의미:** 일부 spectral 계수를 0으로 둘 수 있지만 token, head, weight shape을 없애는 배포용 pruning은 아니다. 같은 크기의 추론 경로에서 **동작을 바꾸는 weight edit**다.
