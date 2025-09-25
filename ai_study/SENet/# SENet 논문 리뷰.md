# SENet 논문 리뷰

## “Squeeze-and-Excitation Networks”

> 정보의 축약을 통해 중요한 특징에 더 집중하는 self-attention mechanism이 녹아들어가 있다. 기존의 CNN 보다 더 *‘명시적’* 이라는 것이 주요 차이점이다.
> 
- 기존 Network에 Add-On 방식으로 사용 가능한 *SE Block 제안*
- Main Flow: **GAP**하고 **FCL** 통과하여 기존 Feature_map **Recalibration**

---

## 1. Introduction

- Channel 간의 관계에 집중
- *Squeeze-and-Excitation (SE) block*을 통해서 구현
- 장점
    1. 간단히 SE block을 쌓는 방식
    2. 기존의 block을 대체하여 사용 가능
    3. 연산/자원 비용의 감소와 동시에 모델의 성능 향상
- Main Structure
    
    ![image.png](attachment:6463ad6b-715d-4aab-b40c-d17f6d586951:image.png)
    
    $X = Input\ data ,\ (H', W', C') = Input\ shape, \ F_{tr} = module\ of\ the\ model \\
    
    U = feature \ map,\ (H, W, C) = shape\ of\ feature\ map  \\
    
    F_{sq} = Squeeze\ block,\ F_{ex} = Excitation\ block,\ F_{scale}=scaler$
    
- *self-gating mechanism*: 자신으로부터 뽑은 요약 벡터를 다시 자신의 채널을 얼마나 증감 시킬지 정하는데 사용하는 메커니즘.

## 2. Squeeze-And-Excitation Blocks

- 지금까지 많은 연구들은 모델과 연산/자원 비용을 줄이는 것에 초점이 맞춰져 있었다.
- ***Squeeze: Global Information Embedding***
    - 기존의 feature map은 작은 영역(local receptive filed)만 보기 때문에 전 문맥을 직접적으로 사용하기 어렵다.
        
        → **공간차원을 축소하여 채널 당 하나의 값으로 요약하여 전역 문맥을 요약**한다. 
        
    - 이때 **Global Average Pooling(GAP)**를 이용한다.
        
        $$
        z_c = \mathrm{F_{sq}}(u_c) = \frac{1}{H W}\sum_{i=1}^{H}\sum_{j=1}^{W} u_c(i,j)
        $$
        
    - Example)
        
        어떤 data가 $(H, W, C) = (2, 2, 2)$ 라면… 
        
        C = 1, $(H, W) = \begin{bmatrix}
        1 & 3 \\
        5 & 7
        \end{bmatrix}$ → ‘GAP’ → $(1+3+5+7)/4 = 4$ (Scalar Value)
        
        C = 2, $(H, W) = \begin{bmatrix}
        2 & 4 \\
        6 & 8
        \end{bmatrix}$ → ‘GAP’ → $(2+4+6+8)/4 = 5$ (Scalar Value)
        
        ---
        
        ⇒ $최종\ shape = (1,1,2)$
        

- ***Excitation Adaptive Recalibration***
    - 앞서 Squeeze로 얻은 $z_c$(채널 요약)을 Input으로 받아 Channel 간 **비선형 의존성을 학습**하고 그 결과를 **(0, 1) 범위의 게이트로 변환**하여 원래의 feature map을 Channel별로 **재보정**한다.
    - 두 개의 FC Layer와 두 개의 activation function으로 이루어져 있다.
        - 비선형 함수: 단순 선형 합(MLP)로는 비선형 관계 확보 불가
        - sigmoid: 여러 채널이 동시에 중요할 수 있음. Softmax를 안쓰는 이유
        - bottleneck: parameter, FLOPs 줄이고 Overfitting 억제 효과
            
            > 차원 축소 → 비선형 → 다시 확장 → 채널별 게이트 → 재보정
            > 
        
        $$
        s = \sigma\!\big(W_2\,\mathrm{ReLU}(W_1 z)\big),\qquad\tilde{u}_c = s_c \cdot u_c
        $$
        
        ![image.png](attachment:657c66e9-f93b-4053-b132-b9024218fa8c:image.png)
        
    - 위의 Main Structure에서 알 수 있듯이 모델의 한 모듈이 끝날 때마다 SE-block이 삽입되어 있다.
    - Channel 수를 보면 256 → 16 → 256, 512 → 32 → 512,… 등 *Bottleneck 구조*를 확인할 수 있고 X16의 비율로 증감한다. 해당 수치는 *Reduction ratio*라는 hyperparmeter로 실험 결과는 아래와 같다.
        
        ![image.png](attachment:510e6cb0-0578-4980-8325-5ce78bb4f551:image.png)
        
    - Exciation은 Layer의 깊이에 따라 하는 역할이 다르다.
        
        ![image.png](attachment:cd25a2eb-0dd7-4ef8-9821-6561f7fd4c9e:image.png)
        
- 결국, $z$는 입력별 요약이고 $s$도 입력 조건화된 가중치이므로 채널 축으로 self-attention한 것으로도 볼 수 있다.

## 3. Points of Confusion

### 1. GAP의 $z$ VS 중요도 $s$

- $z$는 Channel별 전역적 평균 **강도**(활성)이고, $s$는 $z$를 Excitation으로 mapping한 **중요도**이므로 Channel별 **$z$ 값들끼리 *‘크기를 비교한다’*는 것은 불가능**

### 2. 그럼 $z$값은 어떻게 해석?

- 절대값의 의미는 약하다. → Layer, Channel마다 Scale이 다르다.
- 같은 Layer, 같은 Sampled이면 상대 순위 & 정규화(z-score)로 본다.

### 3. Excitation이 non-linear gating인 이유

- ReLU + Sigmoid로 **비선형** 학습 + (0, 1) 범위 **게이팅**

### 4. Excitation에서 FC layer가 하는 역할(CNN 공통)

- Channel별로 얻은 특징들을 한번에 종합해서 보기 위함으로도 볼 수 있다.

### 5. SE block에서 GAP의 역할은?

- CNN은 지역적 특징에 치우쳐 전역 문맥 파악이 어렵기 때문에 SE block의 **GAP가 Channel의 전역 정보를 요약하여 모델에 *‘명시적으로’* 적용**하는 역할을 한다.
    
    (대표값으로서 평균값을 사용한다. → GMP < GAP(실험결과))
    
- 이 요약된 정보는 이후 Recalibration의 근거 신호로 활용된다.

### 6. SE block의 동작 시점과 기준점

- 각 Network Module의 출력 직후 동작한다. Feature map을 Input으로 받기 때문.
- 1개의 data는 1개의 SE block을 1번만 통과한다. (SE block이 여러개면 각 블록을 한 번씩 통과함)

### 7. SE는 일종의 증폭기 역할을 하는데 불균형 데이터에서 일반화 성능이 더 약해질 수 있지 않을까?

- SE는 입력별로 Channel 중요도를 재가중하는 모듈일 뿐, 데이터 분포의 근본 문제를 해결하거나 본질적으로 악화시키는 구조가 아니다.
- 위의 6.의 기준점에 대한 개념을 이해하면 혼동되지 않을 것.