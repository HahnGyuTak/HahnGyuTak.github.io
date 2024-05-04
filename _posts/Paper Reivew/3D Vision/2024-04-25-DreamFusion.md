---
title: "[Paper Review] DreamFusion"
date: 2024-04-30 20:02:43 +/-0000
categories: [Paper Review, 3D Vision, Generative model]
tags: [AI, generative, Diffusion, 3D]   
use_math: true  # TAG names should always be lowercase
typora-root-url: ../../../
---



# **DreamFusion 논문 리뷰**

DreamFusion: Text-to-3D Using 2D Diffusion: [arxiv](https://arxiv.org/abs/2209.14988) 



DreamFusion은 3D Generative model 관련 논문에서 많은 인용수를 자랑한다. 2D 이미지 여러장을 이용해 획기적인 방법으로 3D modeling을 수행하는 Nerf와 2D Diffusion 모델을 잘 섞어서 text-to-3D generation을 수행하는 DreamFusion은 이후 DreamBooth3D, SweetDreamer 등 3D generation 분야에서 SOTA를 달성한 많은 논문들에서 언급된다.



## **Introduction**

지금까지 생성모델은 

