# Optimizers (옵티마이저)

> "어떻게 고칠지" 결정하는 알고리즘  
> Loss를 줄이는 방향으로 가중치를 업데이트

---

## 1. SGD (Stochastic Gradient Descent)

```
w = w - lr × ∂L/∂w
```

```python
torch.optim.SGD(model.parameters(), lr=0.01)
```

- 가장 기본적인 방법
- 기울기 방향으로 lr만큼 이동
- 단점: lr 설정이 까다롭고 수렴이 느릴 수 있음

---

## 2. SGD + Momentum

```
v = β × v - lr × ∂L/∂w
w = w + v
```

```python
torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

- 이전 이동 방향을 기억해서 관성처럼 활용
- 평평한 구간에서도 빠르게 진행

```
비유: 공이 언덕을 굴러내려갈 때
  SGD      → 매 순간 경사만 보고 이동
  Momentum → 이전 속도를 유지하며 이동
```

---

## 3. Adam (Adaptive Moment Estimation)

```
m = β1 × m + (1-β1) × g        ← 1차 모멘트 (기울기 평균)
v = β2 × v + (1-β2) × g²       ← 2차 모멘트 (기울기 분산)

m̂ = m / (1-β1^t)               ← 편향 보정
v̂ = v / (1-β2^t)

w = w - lr × m̂ / (√v̂ + ε)
```

```python
torch.optim.Adam(model.parameters(), lr=0.001)
```

- 각 파라미터마다 lr을 자동으로 조절
- 자주 업데이트된 파라미터 → lr 감소
- 드물게 업데이트된 파라미터 → lr 증가
- 현재 가장 많이 쓰이는 옵티마이저
- 언제: 대부분의 딥러닝 문제에 기본값으로 사용

---

## 4. AdamW

```python
torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

- Adam + Weight Decay (가중치 규제)
- 모델이 너무 복잡해지는 것을 방지 (과적합 방지)
- 언제: Transformer 계열 모델 (GPT, BERT 등)

---

## 전체 비교

| | SGD | Momentum | Adam | AdamW |
|---|---|---|---|---|
| 속도 | 느림 | 중간 | 빠름 | 빠름 |
| lr 설정 | 까다로움 | 까다로움 | 쉬움 | 쉬움 |
| 메모리 | 적음 | 적음 | 많음 | 많음 |
| 주 용도 | 기초 학습 | 이미지 | 대부분 | Transformer |

---

## RIRO Lab 연구와 연결

```
선형 회귀        → SGD
MNIST            → Adam
Behavior Cloning → Adam
DiSPo            → Adam
                   (대부분의 딥러닝 연구에서 기본값)
```

---

## 참고

- 손실함수: [loss_functions.md](./loss_functions.md)
