---
title: "[Paper Review] 3D Gaussian Splatting"
date: 2024-12-16 20:02:43 +/-0000
categories: [Paper Review, 3D Vision, SIGGRAPH2023]
tags: [SIGGRAPH, 3D Vision, Gaussian Splatting]   
use_math: true  # TAG names should always be lowercase
typora-root-url: ../../../
---

# **[논문 리뷰] 3D Gaussian Splatting for Real-Time Radiance Field Rendering**

> **3D Gaussian Splatting for Real-Time Radiance Field Rendering**
>
> Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis
>
> SIGGRAPH 2023.
>
> [[Arxiv](https://arxiv.org/abs/2308.04079)] [[Project Page](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)] [[Github](https://github.com/graphdeco-inria/gaussian-splatting)]



가우시안 스플래팅 -> 3D 가우시안 점(타원모양)을 흩뿌린다



![스크린샷 2024-12-11 오후 5.15.29](/assets/img/GaussianSplatting/GS.png)

## **Introduction**

3D scene 표현에서는 Mesh와 Point 기반 방법들이 GPU환경에 적합하기 때문에 흔히 사용된다. 이와는 다르게 NeRF는 continuous scene 표현을 기반으로 하며, 캡처된 장면의 새로운 view 합성을 위해 volumetric ray-marching을 사용하여 MLP을 최적화한다. 현재까지 가장 효율적인 radiance field 솔루션은 voxel 또는 hash grid 또는 point에 저장된 값을 보간하여 continuous한 representation을 기반으로 구축한다. 이러한 continuous한 방법들은 optimization에 도움되지만, **확률적인(stochastic) sampling에서 많은 비용이 요구되고 noise가 발생할 수 있다.**

저자들이 소개하는 **3D Gaussian representation**은 visual quality SOTA를 달성하고 training 시간을 단축할 수 있다. 또한 tile 기반 splatting 방식은 1080p 해상도에서 SOTA 품질의 실시간 렌더링을 보장한다.

이전까지의 SOTA인 Mip-NeRF360은 최대 48시간의 학습이 필요하며, 최근 method들은 학습 시간을 단축할 수 있었지만 Mip-NeRF360의 visual quality를 따라가지 못한다.

3D Gaussian Splatting은 3가지 주요 구성 요소를 기반으로 한다.

1. **3D Gaussian** 

   * **SfM process에서 제공된 sparse point cloud로 3D Gaussian 집합을 초기화**한다.
   * Multi-View Stereo data가 필요한 기존 point-based 방법들과 달리, SfM point의 입력만으로도 high quality를 얻을 수 있다.

   >  NeRF와 유사한 입력을 사용함에도 불구하고, 특히 ‘NeRF-synthetic dataset’에 대해서는 무작위 초기화로도 높은 품질의 결과를 달성한다. 3D Gaussian은 미분 가능한 volumetric representation이기 때문에, gradient를 사용하여 optimization할 수 있으며, 효율적으로 3D to 2D의 rasterization을 수행할 수 있다. 또한 NeRF와 비슷하게 standard 𝛼-blending을 적용함으로써, NeRF에서 사용되는 것과 동일한 이미지 형성 모델을 사용할 수 있다.

   

2. **Optimization of the properties of the 3D Gaussians** : 3D position, Opacity 𝛼, Anisotropic covariance, and Spherical Harmonic (SH) coefficients
   * optimization 단계에서 3D Gaussian을 제거하거나 추가하는  adaptive density control steps을 수행한다.
3. **실시간 Rendering** 솔루션
   * 빠른 GPU 정렬 알고리즘과 tile-based rasterization에서 영감을 얻었다.
   * [[Pulsar: Efficient Sphere-Based Neural Rendering](https://arxiv.org/abs/2004.07484)]을 참고





## **Related works**

### **과거 Scene Reconstruction and Rendering**

 **Structure-from-Motion (SfM)**: 2006년 SfM는 사진 모음을 사용하여 새로운 시점을 합성할 수 있는 새로운 영역을 가능하게 하였다. SfM은 카메라 보정 중에 sparse point cloud를 추정하며, 초기에는 3D 공간의 간단한 시각화에 사용되었다.

 **다중 시점 스테레오(Multi-View Stereo, MVS)**: MVS 기술의 impressive full 3D reconstruct 알고리즘은 입력 이미지를 새로운 시점 카메라로 재투영하고 혼합하여, 이 기하학을 사용하여 재투영을 안내하는 여러 view synthesis 알고리즘의 개발을 가능하게 했다.

하지만, MVS가 존재하지 않는 unreconstruct or over-reconstruction 영역을 커버할 수는 없다.



### **Neural Rendering and NeRF**

딥러닝 기술이 View synthetic에 도입이 되었지만 여전히 MVS 기반 geometry를 사용한다는 단점이 있다. 또한 CNN을 최종 렌더링에 사용하는 것은 temporal flickering이 발생한다.

> temporal flickering : 시각적 콘텐츠에서 개별 프레임들이 연속성을 유지하지 못하고, 시각적으로 불안정하거나 불규칙하게 보일 때 나타는 현상

**Volumetric representations**은 3차원 공간의 부피를 전체적으로 모델링하는 기법으로, Soft3D에 의해 시작되었다. 이후 geometry를 표현하기 위해 continuous하고 미분 가능한 density field를 기반으로 딥러닝과 Volumetric ray-marching을 결합하였다. Volumetric ray-marching을 사용하는 렌더링은 필요한 sample 수가 많기 때문에 많은 cost가 소요된다.

![스크린샷 2024-12-12 오전 4.52.34](/assets/img/GaussianSplatting/nerf.png)

**<u>NeRF</u>**는 다양한 각도에서 촬영된 이미지와 해당 이미지의 카메라 파라미터를 NeRF 모델에 입력하면, 특정 연산을 통해 새로운 뷰를 출력해준다. 이를 위해 투명도를 나타내는 $\alpha$가 도입되었다. MLP로 구성되어있는 NeRF는 optimization을 위해 Positional Encoding과 Hierarchical Sampling을 사용하였지만, 대규모 MLP 구조이기 때문에 training 속도에 부정적인 영향을 끼쳤다. 현재 novel view synthetic 분야 SOTA는 Mip-NeRF360으로, 렌더링 품질을 뛰어나지만 train 및 렌더링 시간은 여전히 매우 오래 걸린다.

이후 나온 method중 주목할 만한 것은 <u>InstantNGP</u>(hash grid와 occupancy grid를 사용하여 계산을 가속화하고 더 작은 MLP를 사용하여 density와 모양을 표현), <u>Plenoxels</u>(sparse voxel grid를 사용하여 continuous density field를 보간하고 NN을 사용하지 않음)이다. 이 두 방법 모두 빈 공간을 효과적으로 표현하는데 어려움을 겪는다. 또한 grid의 선택에 따라 이미지 품질이 제한되며 ray-marching step에 따라 많은 sample을 처리해야하기 때문에 렌더링 속도가 저하된다.



### **Point-Based Rendering**

Point-based 렌더링은 연결되지 않은 unstructured geometry sample인 **포인트 클라우드**를 효율적으로 렌더링하는 방법이다. 최근 연구들은 원형 또는 타원 디스크, 타원체 등 픽셀값보다 큰 범위의 포인트 요소를 splatting한다.

NeRF와 Point-based $\alpha$-blending은 본질적으로 동일한 이미지 형성 모델을 사용한다.

렌더링된 pixel의 색상인 $C$는 ray(광선)을 따라 volumetric 렌더링을 통해 다음과 같이 얻어진다.
$$
C = \sum_{i=1}^N T_i (1 - \exp(-\delta_i \sigma_i)) c_i \quad \text{with} \quad T_i = \exp\left(-\sum_{j=1}^{i-1} \delta_j \sigma_j\right)
$$
$\sigma$ : density(밀도), $T$ : transmittance(투과율), $c$ : color, $\delta$ : 간격

위 식은 불투명도를 의미하는 $\alpha$를 통해 다음과 같이 된다.
$$
C = \sum_{i=1}^{N}T_i\alpha_i c_i, \; \alpha_i = 1-\text{exp}(-\sigma_i\delta_i), \; T_i = \prod_{j=1}^{i-1}(1-\alpha_j)
$$


일반적으로 neural point-based 방식에서도 위 식을 따라 픽셀 색상을 계산한다.

하지만, Nerf와 Point-based 방식의 렌더링 알고리즘은 완전히 다르다.

NeRF는 비어있거나 채워진 공간을 implicit하게 나타내는 continuous representation이다. 위 식에서 sample을 찾으려면 cost가 큰 random sampling과정이 필요하고, noise가 발생한다.

반면, Point는 구조화되지않은 discrete representation이다. NeRF처럼 geometry의 생성, 이동, 파괴가 가능한 만큼 유연성이 있다. 이는 full volumetric representation의 단점을 피하면서 불투명도(opacity)와 위치(position)을 optimizing한다. Point-based의 최근 기술들은 MVS를 필요로 하지않고 Spherical Harmonics으로 방향을 지정하지만, 단일 객체 장면만 처리할 수 있고, 초기화를 위해 mask가 필요하다.



저자들은 **3D Gaussian을 사용하여 더 유연한 Scene representation을 가능하게 한다.** MVS가 필요없으며 tile-based 렌더링 알고리즘을 사용하여 실시간 렌더링을 수행한다.

인체 capture 분야에서는 이미 3D Gaussian으로 인체를 표현하는데 사용되었지만, 특정 인체 부위나 얼굴을 렌더링하는 특정 사례에 초점을 맞춰 사용되고 있기 때문에, 작은 depth complexity를 지닌 scen을 생성하는 데 특화되어있다.

저자들은 이에 영감을 얻어 **anisotropic covariance의 optimizing**, **interleaved optimization/density 제어**, **depth 정렬**을 통해 복잡하고 depth complexity가 큰 장면을 처리할 수 잇다.





## **Overview**

![스크린샷 2024-12-13 오전 4.14.26](/assets/img/GaussianSplatting/overview.png)

흐름을 간단히 짚고 넘어가보자.

1. SfM으로 얻은 sparse point cloud에서 초기 3D 가우시안의 위치($\mu$), 공분산 $\Sigma$, 불투명도 $\alpha$를 초기화한다.
2. 3D Gaussian을 Projection하여 2D Gaussian 형태가 된다. 이를 나중에 Ground truth와 비교하여 parameter를 업데이트한다.
3. 미분가능한 형태의 Tile-based Rasterization을 통해 2D Gaussian들을 하나의 이미지로 생성한다.
4. 생성된 이미지와 Ground truth 이미지의 Loss를 계산하고 gradient를 얻는다.
5. gradient를 기반으로 Adaptive Density Control을 통해 3D Gaussian의 형태를 변화시킨다

> 제안된 방법의 input은 정적 scene의 이미지 집합과 SfM에 의해 보정된 카메라이다. 이 input으로 sparse point cloud를 생성하며 포인트들로부터 position(mean), 공분산 행렬(covariance matrix), 불투명도(opacity) $\alpha$를 정의하는 3D Gaussian 집합을 생성한다. 이 3D Gaussian은 높은 anisotropic Volumetic splats을 사용하여 3D scene의 간결한 표현을 가능하게 한다. Radiance field의 색상은 Spherical harmonics를 통해 표현된다. Gaussian density의 adaptive한 제어, 3D Gaussian parameter(위치, 공분산, $\alpha$, SH coefficients)의 일련의 optimization을 통해 Radiance field를 생성한다. 이 방법에서 효율성은 tile-based Rasterizer가 담당하는데, 빠른 sorting을 통해 anisotropic splat의 $\alpha$-blending을 허용하고 가시성 순서를 고려한다.

> SH coefficients : 빛의 방향에 따른 색상 변화를 조절





## **Differentiable 3D Gaussian Splatting**

SfM 기법을 통해 얻은 희소 point cloud 데이터는 위치정보는 있지만 표면 방향 정보(normal)는 포함하고 있지 않다. 이 input을 통해 high-quality의 Novel View Synthesis의 Scene representation을 얻는 것이 목표이다. 이를 위해서 미분 가능한 volumetric representation이지만 구조화되지 않고 빠른 렌더링이 가능한 3D Gaussian을 선택하였다. 2D splats에 쉽게 투영할 수 있어 렌더링을 위한 빠른 $\alpha$-blending이 가능하다. 

기존 방법들은 2D point를 사용하며 각 점이 법선(normal)을 가진 작은 평면 원이라고 가정한다. SfM의 point는 드문 희소성을 가지므로 normal을 추정하는 것은 한계가 있으며 추정하더라도 noise가 많은 법선을 최적화하는 것이 어렵다. 

3D Gaussian은 normal이 필요가 없고, Point $\mu$(평균)를 중심으로 전체 3D 공분산 행렬 $\Sigma$로 정의된다.

> $\mu$로 위치가 정해지고, $\Sigma$로 형태와 크기가 정해짐

$$
G = e^{-\frac{1}{2}(x)^T \Sigma^{-1}(x)}
$$

Blending process에서 이 $G$는 Opacity $\alpha$와 곱해진다. 



Rendering을 위해서는 3D Gaussian을 2D로 투영해야한다. Viewing transformation $W$가 주어졌을 때, 이미지 좌표의 공분산 행렬 $\Sigma^`$는 다음과 같이 얻어진다.
$$
\Sigma^` = JW \; \Sigma \; W^TJ^T
$$

> $W$ :  World좌표계 -> Camera 좌표계 변환 matrix (Viewing Transformation)
> $J$ : Projective Transformation의 선형 근사치의 Jacobian matrix
>
> $\Sigma$ : World 좌표계에서의 공분산 행렬 (3D)
> $\Sigma^`$ : Image 좌표계에서의 공분산 행렬 (2D)



이론적으로는 공분산 행렬 $\Sigma$를 직접 최적화하여 3D Gaussian을 형성하고자 하는 것이 직관적인 접근이다. 이렇게 하면 Radiance field를 표현하는 데 필요한 3D Gaussian의 형태를 조정할 수 있다. 하지만, 공분산 행렬은 물리적 의미를 갖기 위해서는 반드시 양의 준정부호(positive semi-definite)이어야 한다. 

>  양의 준정부호란 모든 고유값이 0 또는 양수인 행렬을 의미하며, 이는 통계적으로 분산이 음수가 될 수 없음을 보장한다.

하지만, gradient descent을 사용하여 모든 parameter를 최적화할 때, 이러한 제약 조건을 쉽게 만족시킬 수 없다. (공분산 행렬이 음수가 될 수도 있음) 따라서 매개변수의 업데이트나 경사 계산 과정에서 유효하지 않은(즉, 양의 준정부호가 아닌) 공분산 행렬이 생성될 수 있다.

3D Gaussian의 공분산 행렬 $\Sigma$는 타원체로 설명하는것과 매우 유사하다. Scaling matrix $S$와 Rotation matrix $R$이 주어지면 다음을 통해 $\Sigma$를 구할 수 있다.

> Scailing matrix라 했지만 결국 3D Gaussian의 공분산 행렬이다

$$
\Sigma = RSS^TR^T
$$

두 요소 $R$, $S$ 를 독립적으로 optimization하기 위해, scaling을 위한 3D vector $s$와 회전을 나타내는 quaternion(사원수) $q$ 를 별도로 저장한다. 

> quaternion은 4개의 수 $(x, y, z, w)$로 이루어지며 하나의 벡터 $(x,y,z)$와 스칼라값 $w$로 구성된다. 오일러 각과 다르게 세 축을 동시에 회전시키긴다. 이는 방향(orientation)과 회전(rotation) 둘을 다 표현할 수 있다.
>
> 하지만 quaternion의 회전은 한 orientation 에서 다른 orientation 으로 측정하기에 180 보다 큰 값을 표현할 수 없다는 단점이 있다. 

이를 각각의 행렬로 간단히 변환하고 결합하여 유효한 단위 quaternion을 얻기 위해 𝑞를 정규화할 수 있다.
훈련 중 자동 미분으로 인한 상당한 오버헤드를 피하기 위해 모든 파라미터에 대한 기울기를 명시적으로 도출한다. 정확한 도함수 계산에 대한 자세한 내용은 부록 A에 나와 있다.
연산에 적합한 anisotropic 공분산의 표현을 통해 캡처된 장면에서 다양한 모양의 기하학적 구조에 적응하도록 3D Gaussian 모델을 최적화하여 간결한 표현을 얻을 수 있다. 그림 3은 이러한 경우를 보여준다.





## **Optimization with Adaptive Density Control of 3D Gaussians**

Method의 핵심인 Optimization이다. 최종적으로 Free-view 합성을 위하여 scene을 정확하게 표현하는 high-density 3D Gaussian 집합을 생성한다.



### **1. Optimization**

학습은 렌더링과 데이터셋의 Ground truth와 결과 이미지의 비교를 반복하며 3D Gaussian을 최적화한다.

최적화 과정에서 geometry를 생성, 제거, 이동하며 최적의 geometry를 찾아야한다.

특징은 다음과 같다

* Stochastic Gradient Descent(SGD)를 사용

* Opacity $\alpha$ : $[0 , 1)$ 범위로 제한, Sigmoid 활성화 함수

* 공분산 $\Sigma$ : 지수 활성화 함수, 가까운 3 지점과의 거리의 평균과 같은 축을 가진 isotropic Gaussian으로 초기화 

* Loss Function : D-SSIM이 결합된 L1 loss,  ($\lambda$는 0.2)
  $$
  \mathcal{L} = (1-\lambda)\mathcal{L}_1 + \lambda\mathcal{L}_{\text{D-SSIM}}
  $$
  

### **2. Adaptive Control of Gaussians**

SfM의 초기 sparse point 집합으로 시작한 다음, 단위 volume에 대해 Gaussian 수와 density를 적응적으로 제어하는 방법을 적용한다.

Optimization Warm-up을 수행한 이후, 100회 반복할 때마다 불투명도 $\alpha$가 임계값 $\epsilon$보다 작은 Gaussian을 제거한다.

![스크린샷 2024-12-13 오전 5.33.29](/assets/img/GaussianSplatting/adaptive.png)

Adaptive control은 다음 2가지에 초점을 맞춰 Gaussian 이동을 시도한다.

1. Under reconstruction
   * 기하학적 특징이 누락된 영역
   * 동일한 크기의 복사본을 생성하고 위치 gradient 방향으로 이동
   * 총 부피, Gaussian의 수 모두 증가
2. Over reconstruction
   * Gaussian들이 scene의 넓은 영역을 덮고 있는 영역
   * 더 작은 Gaussian으로 분할하고 sampling을 위해 원본 3D Gaussian PDF를 사용하여 위치를 초기화한다
   * 총 부피 보존, Gaussian 수 증가



다른 Volumetric representation처럼 입력 카메라에 가까운 floater로 인해 최적화가 멈출 수 있으며, 이 경우 Gaussian density가 부당하게 증가될 수 있다. 

Gaussian 수 증가를 완화하는 방법은 3000iter 마다 $\alpha$를 0으로 설정한다. 최적화가 필요없는, 즉 많이 겹치는 Gaussian을 제거하며 Gaussian수를 제어할 수 있다. 



## **Fast Differentiable Rasterizer for Gaussians**

효율적인 optimizing을 위해 $16 \times 16$ tile로 나누는 Tile-based Rasterization을 도입한다. 분할한 타일에 대하여 Culling을 진행하여 카메라 view에 보이는 Gaussian만을 유지한다. 남은 Gaussian들은 view space depth와 tile ID를 결합한 Key를 기준으로 정렬한다. 

초기 정렬을 기반으로 $\alpha$-blending이 이루어지며, Rasterization는 blending된 Gaussian에만 역전파를 하기 때문에 추가 메모리 소비가 낮다. 렌더링을 위해 각 타일에 하나의 thread 블록이 실행되며, 주어진 픽셀에 대해 $c$와 $\alpha$값을 누적한다.



## **Evaluation**	

![스크린샷 2024-12-13 오후 8.21.04](/assets/img/GaussianSplatting/eval1.png)

![스크린샷 2024-12-13 오후 8.27.18](/assets/img/GaussianSplatting/table1.png)

비교 벤치마크는 이전까지의 SOTA인 Mip-NeRF360과 최근 2가지 고속 NeRF인 InstantNGP, Plenoxels으로 한다.

metric은 SSIM, PSNR, LPIPS로 하였으며, 학습시간, 렌더링 속도, Memory의 측면에서도 비교한다.



3D Gaussian Splatting을 Instant-NGP와 비교하였을 때 Train속도가 비슷하지만, PSNR이 높고, 랜더링 속도에서 큰 향상을 보인다.

> $7K$, $30K$는 iteration을 의미하며, 아래 그림을 보면 $7K$에서 이미 좋은 품질을 보이고 있음을 확인할 수 있다.
>
> <img src="/assets/img/GaussianSplatting/eval2.png" alt="스크린샷 2024-12-13 오후 8.30.56" style="zoom:70%;" />



### **Ablations**

**Initialization from SfM**

SfM point cloud로 Gaussian의 초기화에 대한 비교를 진행한다.

![스크린샷 2024-12-13 오후 9.33.24](/assets/img/GaussianSplatting/ablations1.png)

위 사진에서 Random은 Input camera의 bounding box 크기의 3배 사이즈로 cube를 만들어 균등하게 샘플링한 결과이다.

이 역시 완전한 실패는 아니지만, 배경에서 성능 저하를 확인할 수 있다.



**Densification**

![스크린샷 2024-12-13 오후 9.50.46](/assets/img/GaussianSplatting/ablations2.png)

Adaptive Density control에서 분할 및 복제를 각각 사용한 결과이다.

배경을 잘 구성하려면 Gaussian Split이 중요하지만, 대신 작은 Gaussian을 복제하면 얇은 구조물으로 빠르게 수렴할 수 있다.



**Anisotropic Covariance**

3D Gaussian 전체를 optimization하는 것의 효과를 입증하기 위해 3축에서 3D Gaussian의 반지름을 단일 scalar값으로 바꾸어 Anisotropic을 제거하여 비교하였다.

![스크린샷 2024-12-13 오후 10.05.52](/assets/img/GaussianSplatting/ablations3.png)

Anisotropic을 통해 3D Gaussian에서 surface를 정렬하는 기능이 크게 향상되어 동일한 point 수를 유지하면서 렌더링 품질을 훨씬 더 높일 수 있음을 확인할 수 있다.



## **Limitations**

언급된 limitation은 다음과 같다

* **Sparse Scene에서의 artifact**:

  잘 관찰되지 않는 영역에서는 artifact가 발생할 수 있으며, 이러한 문제는 다른 방법들도 마찬가지로 겪고 있다 (예: Mip-NeRF360). 이는 장면의 특정 부분이 충분히 캡처되지 않았을 때 일반적으로 발생한다.

* **popping artifacts**:

​	Optimizing 과정에서 큰 가우시안을 생성할 때, 특히 View에 따라 모양이 달라지는 영역에서 popping artifacts가 발생할 수 있다. 이는 래스터라이저의 guard band가 Gaussian을 약간 제거하기 때문이다.

* **정규화 부족**:

​	현재 Optimizing 과정에서 정규화를 적용하지 않고 있어, Sparse Scene과 popping artifacts 문제를 완화하는 데 도움이 될 수 있다.

* **메모리 사용량**:

​	NeRF 기반 솔루션에 비해 상당히 높은 메모리 사용량을 기록하며, 특히 큰 장면을 훈련할 때 GPU 메모리 사용량이 20GB를 초과할 수 있다. 그러나 최적화된 저수준 구현을 통해 이러한 수치를 크게 줄일 수 있다.