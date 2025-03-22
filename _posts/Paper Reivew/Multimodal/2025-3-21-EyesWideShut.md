---
title: "[Paper Review] Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs"
date: 2025-03-21 18:21:01 +/-0000
categories: [Paper Review, Multi Modal]
tags: [Multi-modal, CLIP]   
use_math: true  # TAG names should always be lowercase
typora-root-url: ../../../
---

# **[논문 리뷰] Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs**

> **Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs**
>
> [Shengbang Tong](https://arxiv.org/search/cs?searchtype=author&query=Tong,+S), [Zhuang Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu,+Z), [Yuexiang Zhai](https://arxiv.org/search/cs?searchtype=author&query=Zhai,+Y), [Yi Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma,+Y), [Yann LeCun](https://arxiv.org/search/cs?searchtype=author&query=LeCun,+Y), [Saining Xie](https://arxiv.org/search/cs?searchtype=author&query=Xie,+S)
>
> CVPR 2024 (Highlight)
>
> [[Arxiv](https://arxiv.org/abs/2401.06209)]

CVPR2024에서 Highlight를 받은 논문이다.

 Multimodal LLMs의 시각적 한계에 대한 연구이며 특히 CLIP과 통합될 때의 한계를 중점적으로 다룬다. 연구는 CLIP가 시각적으로 확연히 다른 이미지를 유사하게 인식하는 “CLIP-blind 쌍”을 식별하는 데 초점을 둔다.



## **Instroduction**

![스크린샷 2025-03-21 오후 7.09.32](/assets/img/2025-3-21-EyesWideShut/figure1.png)

Multimodal LLM(MLLMs)은 최근 급격히 발전해왔지만, 여전히 위와 같이 visual적인 결함이 존재한다. 논문에서는 문제의 원인을 탐구하고자 하며, **visual representations**과 관련이 있다고 제안한다.



저자들은 다음과 같은 연구를 진행한다.

* **CLIP-blind pairs**(CLIP이 비슷하다고 인식하지만 시각적으로는 다른 이미지들)를 사용하여, **MultiModal Visual Patterns(MMVP) 벤치마크를 구성**
* 다양한 CLIP based Vision-Language 모델을 평가하고 CLIP이 어려워하는 시각 패턴과 MLLM의 성능 사이에 상관관계 탐구
* MLLM의 시각적 근거 능력을 상당히 향상시킬 수 있는 *Mixture of Features*(MoF) 접근 방식을 제안

> 대부분의 MLLM은 Vision Encoder로 CLIP을 채택한다. CLIP이 잘 인코딩하지 못하는 경우를 찾고(2장),  \<Mass- producing failures of multimodal systems with language models>에서처럼 Embedding space에서의 *erroneous agreements*를 이용한다. 
>
> 만약 두 이미지가 시각적으로 다르지만 CLIP Vision Encoder에 의해 유사하게 인코딩된다면, 적어도 하나의 이미지는 모호하게 인코딩될 가능성이 있다. 이를 **CLIP-blind pairs**이라고 부르기로 하고, 이미지 간의 시각적 유사성을 측정하기 위해, DINOv2와 같은 vision-only self-supervised encoder를 사용한다. 
>
> 
>
> 저자들은 CLIP-blind pairs가 실제로 downstream MLLM에서 오류로 이어짐을 발견하였고, MultiModal Visual Patterns 벤치마크를 도입한다. 또한 MLLM의 실패 사례를 찾고, MMVP에서 CLIP이 어려워하는 시각 패턴을 탐구하고(3장), 대규모 CLIP 기반 모델로 이를 완화할 수 있는지 여부를 평가한다. 
>
> 9개의 확인된 패턴 중 7개가 어떠한 대규모 CLIP 기반 모델로도 해결될 수 없음을 발견한다. 게다가, CLIP이 특정 패턴, 예를 들어 **방향**에 어려움을 겪는 경우, MLLM도 비슷한 단점을 보일 가능성이 높다는 것을 발견한다. 이는 CLIP 시각 인코더가 이러한 시스템에서 병목 현상이 될 수 있음을 보여준다.
>
> 
>
> 마지막으로, 우리는 MLLM의 시각적 근거를 개선하기 위한 단계를 밟는다. MLLM의 시각적 결함이 CLIP 모델에 의존하기 때문에, 우리는 시각 중심 표현을 MLLM에 통합하는 영향을 조사한다(4장). 구체적으로, 우리는 오직 시각적 자기지도 모델인 DINOv2 [42]와 같은 시각 전용 자기지도 모델을 통합하는 방법을 탐구한다. 우리는 이 기술을 ‘특징 혼합’(MoF)이라고 부른다. 우선, 우리는 CLIP 및 DINOv2 기능을 다양한 비율로 선형 혼합하는 것을 참조로서 첨가-MoF(A-MoF)를 도입한다. 이 과정은 DINOv2 기능이 시각적 근거에 더 효과적이지만, 지시를 따르는 능력이 감소하는 비용을 초래함을 밝힌다. 이를 해결하기 위해, 우리는 CLIP 및 DINOv2 모델의 시각적



## **MMVP Benchmark**

![fig2](/assets/img/2025-3-21-EyesWideShut/fig2.png)

### **1. CLIP-blind Pairs**

CLIP vision encoder가 잘 인코딩하지 못하는 이미지를 직접 찾기 위해, 우리는 [Tong et al.](https://arxiv.org/abs/2306.12105)에서 제안된 아이디어를 확장하여 Vision model에서 blind 쌍을 자동으로 찾는다. 

기본 원리는 간단하다: 

1. 두 이미지가 눈에 띄게 시각적으로 다르지만 CLIP vision encoder에 의해 유사하게 인코딩된다면, 그 중 하나는 모호하게 인코딩될 것이다(figure 왼쪽).

2. 두 이미지 간의 시각적 차이를 측정하기 위해, 우리는 reference 모델 내에서 이미지 representations을 검토한다:

   > vision- only self-supervised model인 **DINOv2**

ImageNet과 LAION-Aesthetics에서 CLIP-blind pairs를 수집한다.

CLIP-ViT-L-14 를 사용하여 embedding을 계산하고 DINOv2-ViT-L-14를 사용하여 DINOv2 embedding을 계산하여 cosine similiarity가 CLIP은 0.95 이상이고 DINOv2은 0.6 미만인 pairs을 수집한다.



### **2. Designing Benchmark from CLIP-blind Pairs**

Multimodal Visual Patterns(MMVP) 벤치마크를 소개하고 Visual Question Answering (VQA) 벤치마크를 사용한다. 

수집된 CLIP-blind pairs를 활용하여, 150쌍의 이미지와 300개의 질문을 신중하게 디자인한다. 각 pair에 대해, 우리는 CLIP vision encoder가 간과하는 detail을 직접 찾고(위 그림 중앙), 이러한 detail을 탐구하는 질문을 만든다. 

>  예를 들어 “Is the dog facing left or right?”(위 figure 오른쪽, 아래 figure). 주 목적은 MLLM 모델이 비교적 간단하고 명확한 질문에 실패하고, 결정적인 detail을 간과할 것인지를 결정하는 것이다. 따라서 질문은 단순하고 모호하지 않게 제작된다.

![fig3_benchmark_wide_gemini](/assets/img/2025-3-21-EyesWideShut/fig3_benchmark_wide_gemini.png)

### **3. Benchmark Results**

SOTA 오픈 소스 모델인 LLaVA-1.5, InstructBLIP, Mini-GPT4와 폐쇄 소스 모델인 GPT-4V, Gemini, Bard를 평가하였다. 

평가에서 각 질문은 독립적으로 질문되어 채팅 이력에서의 bias를 제거하였고, user 연구를 통해 인간의 성능도 포함시켰다.

![fig4](/assets/img/2025-3-21-EyesWideShut/fig4.png)

인간과 MLLM 모델 사이에는 상당한 성능 격차가 있다. MLLM은 종종 인상적인 결과를 보여주지만, GPT-4V와 Gemini를 제외한 모델은 Random Guess인 25% 미만 점수를 얻었다. 가장 진보된 GPT-4V와 Gemini도 기본적인 시각적 근거 질문에 도전을 받는다. 그 결과가 Instroduction의 이미지와 2. Designing Benchmark from CLIP-blind Pairs에 있는 이미지이다. 이는 모델의 크기나 훈련 데이터에 관계없이 시각적 detail을 처리하는 데 어려움을 겪는다는 것을 시사한다.



## **Systematic Failures in CLIP**

