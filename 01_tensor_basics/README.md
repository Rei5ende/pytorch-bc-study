# 01. 텐서 기초 + 자동미분 + 선형 회귀

---

## 배운 내용

### 텐서 (Tensor)
PyTorch의 기본 데이터 단위.

```python
torch.zeros(3, 4)   # 영텐서 — 모든 원소가 0
torch.ones(3, 4)    # 일텐서 — 모든 원소가 1
torch.rand(3, 4)    # 랜덤텐서 — 0~1 사이 랜덤값
torch.eye(3)        # 단위행렬 — 대각선만 1, 나머지 0
```

원소별 연산 (+, *) 지원:
```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
a + b  # tensor([5., 7., 9.])
a * b  # tensor([4., 10., 18.])
```

---

### 자동미분 (Autograd)
PyTorch가 연산 과정을 기억하고 기울기를 자동으로 계산하는 기능.

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2        # y = x²
y.backward()      # 역전파
print(x.grad)     # dy/dx = 2x = 6
```

- `requires_grad=True` — 이 텐서의 기울기를 추적하겠다는 선언
- `grad_fn` — PyTorch가 어떤 연산으로 만들어졌는지 기억하는 방식
- `backward()` — 역전파로 기울기 계산

---

### 손실함수 vs 옵티마이저

| | 손실함수 | 옵티마이저 |
|---|---|---|
| 역할 | "얼마나 틀렸는지" 측정 | "어떻게 고칠지" 결정 |
| 출력 | Loss 값 (숫자 하나) | 업데이트된 weight |
| 예시 | MSELoss, CrossEntropyLoss | SGD, Adam |

```python
criterion = nn.MSELoss()                          # 회귀 문제
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

**Learning Rate (lr)**
- 너무 크면 → 발산
- 너무 작으면 → 학습이 너무 느림
- `lr=0.01` → 안정적으로 수렴

---

### 선형 회귀 학습 루프

```python
for epoch in range(500):
    pred = model(x)           # 1. 예측 (forward)
    loss = criterion(pred, y) # 2. loss 계산
    optimizer.zero_grad()     # 3. 기울기 초기화
    loss.backward()           # 4. 역전파
    optimizer.step()          # 5. 가중치 업데이트
```

---

## 실습 결과

`y = 2x + 1` 데이터로 선형 회귀 학습:

```
Epoch 0,   Loss: 94.6359
Epoch 100, Loss: 0.0030
Epoch 200, Loss: 0.0015
Epoch 300, Loss: 0.0008
Epoch 400, Loss: 0.0004

학습된 weight: 2.0092  (정답: 2)
학습된 bias:   0.9669  (정답: 1)
```

Loss가 94 → 0.0004로 수렴하며 정답에 근접.

---

## 다음 단계

[02_neural_network](../02_neural_network/README.md) — MNIST 분류기 (CNN)
