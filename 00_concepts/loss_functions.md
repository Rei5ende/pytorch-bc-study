# Loss Functions (손실함수)

> "얼마나 틀렸는지" 측정하는 함수  
> 모든 딥러닝 학습의 기준점

---

## 1. MSE (Mean Squared Error)

```
L = (1/n) × Σ(y_pred - y_true)²
```

```python
nn.MSELoss()
```

- 오차를 제곱해서 평균
- 큰 오차에 더 큰 패널티
- 언제: 회귀 문제 (연속적인 값 예측)
- 예: 선형 회귀, 가격 예측, DiSPo (연속적인 행동값 예측)

---

## 2. MAE (Mean Absolute Error)

```
L = (1/n) × Σ|y_pred - y_true|
```

```python
nn.L1Loss()
```

- 오차의 절댓값 평균
- MSE보다 이상치(outlier)에 덜 민감
- 언제: 이상치가 많은 데이터의 회귀 문제

---

## 3. Cross Entropy Loss

```
L = -Σ y_true × log(y_pred)
```

```python
nn.CrossEntropyLoss()
```

- 예측 확률 분포와 정답 분포의 차이를 측정
- 확률이 0에 가까울수록 Loss가 급격히 커짐

```
정답: 고양이
예측: 고양이 90% → Loss 작음  (-log(0.9) = 0.10)
예측: 고양이 10% → Loss 큼    (-log(0.1) = 2.30)
```

- 언제: 분류 문제 (MNIST, Behavior Cloning 등)

---

## 4. Binary Cross Entropy

```
L = -(y × log(p) + (1-y) × log(1-p))
```

```python
nn.BCELoss()
```

- Cross Entropy의 이진 분류 버전
- 언제: 참/거짓, 0/1 분류 문제

---

## 언제 어떤 걸 쓸까

| 문제 유형 | 손실함수 | 예시 |
|---|---|---|
| 회귀 | MSELoss | 선형 회귀, DiSPo |
| 다중 분류 | CrossEntropyLoss | MNIST, Behavior Cloning |
| 이진 분류 | BCELoss | 참/거짓 판단 |
| 이상치 많은 회귀 | L1Loss | 노이즈 많은 데이터 |

---

## 참고

- 옵티마이저: [optimizers.md](./optimizers.md)
