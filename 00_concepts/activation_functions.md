# Activation Functions (활성화 함수)

> 신경망에 비선형성을 추가하는 함수
> Linear 레이어만 쌓으면 아무리 깊어도 선형 모델과 같아지는 문제를 해결

---

## 왜 필요한가

```
Linear(4→64) → Linear(64→64) → Linear(64→2)

수학적으로:
  y = W3(W2(W1x))
    = (W3·W2·W1)x
    = Wx  ← 결국 하나의 선형 변환과 같음
```

레이어를 아무리 많이 쌓아도 선형 모델과 동일.  
선형 모델로는 복잡한 패턴 학습 불가능.

활성화 함수를 추가하면:
```
Linear → ReLU → Linear → ReLU → Linear

더 이상 하나의 선형 변환으로 합칠 수 없음
→ 복잡한 패턴 학습 가능
```

---

## 1. ReLU (Rectified Linear Unit)

```
ReLU(x) = max(0, x)

음수 → 0
양수 → 그대로
```

```python
nn.ReLU()
```

**장점:**
- 계산이 단순 (max 연산 하나)
- 기울기 소실 문제가 적음
- 실제 성능이 잘 나옴

**단점:**
- 음수 입력에서 기울기가 0 → Dying ReLU 문제

**언제:** 은닉층 기본값. 대부분의 경우 첫 번째 선택.

---

## 2. Sigmoid

```
σ(x) = 1 / (1 + e^(-x))

출력 범위: 0 ~ 1
```

```python
nn.Sigmoid()
```

**장점:**
- 출력이 0~1 사이 → 확률로 해석 가능

**단점:**
- 기울기 소실 문제 (x가 크거나 작으면 기울기 ≈ 0)
- 학습이 느려짐

**언제:** 이진 분류 출력층 (참/거짓)

---

## 3. Tanh

```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

출력 범위: -1 ~ 1
```

```python
nn.Tanh()
```

**장점:**
- 출력이 -1~1 → 0 중심 (Sigmoid보다 학습 안정적)

**단점:**
- 여전히 기울기 소실 문제 있음

**언제:** RNN 계열 모델 내부

---

## 4. Leaky ReLU

```
LeakyReLU(x) = x       (x > 0)
             = 0.01x   (x ≤ 0)
```

```python
nn.LeakyReLU(negative_slope=0.01)
```

**장점:**
- ReLU의 Dying ReLU 문제 해결
- 음수에서도 작은 기울기 유지

**언제:** ReLU로 학습이 잘 안 될 때

---

## 5. Softmax

```
Softmax(xi) = e^xi / Σe^xj

출력의 합 = 1 → 확률 분포
```

```python
nn.Softmax(dim=1)
```

**장점:**
- 모든 출력값의 합이 1 → 다중 분류 확률로 해석 가능

**단점:**
- CrossEntropyLoss에 이미 포함되어 있어서 출력층에 따로 붙이면 안 됨

**언제:** 다중 분류 출력층 (단, CrossEntropyLoss 사용 시 불필요)

---

## 비교표

| 함수 | 출력 범위 | 기울기 소실 | 주 용도 |
|------|-----------|-------------|---------|
| ReLU | 0 ~ ∞ | 적음 | 은닉층 기본값 |
| Sigmoid | 0 ~ 1 | 있음 | 이진 분류 출력층 |
| Tanh | -1 ~ 1 | 있음 | RNN 내부 |
| Leaky ReLU | -∞ ~ ∞ | 없음 | ReLU 대체 |
| Softmax | 0 ~ 1 | - | 다중 분류 출력층 |

---

## 이 프로젝트에서의 사용

```
MNIST 분류기:
  Linear → ReLU → Linear → ReLU → Linear
  (출력층에 ReLU 없음 — CrossEntropyLoss가 처리)

Behavior Cloning:
  Linear(4→64) → ReLU → Linear(64→64) → ReLU → Linear(64→2)
  (동일한 이유로 출력층에 ReLU 없음)
```

---

## 참고

- 손실함수: [loss_functions.md](./loss_functions.md)
- MNIST 모델: [../02_neural_network/README.md](../02_neural_network/README.md)
- BC 모델: [../04_behavior_cloning/README.md](../04_behavior_cloning/README.md)
