# 02. 신경망 구현 (MNIST 분류기)

## MNIST란?

**Modified National Institute of Standards and Technology**의 약자.  
미국 국립표준기술연구소(NIST)가 수집한 손글씨 숫자 데이터셋을 수정(Modified)한 것.

- 학습 데이터: 60,000개
- 테스트 데이터: 10,000개
- 이미지 크기: 28×28 픽셀 (흑백)
- 클래스: 0~9 숫자 10가지

---

## 전체 기술 흐름

```
원본 이미지 (28×28 픽셀)
        ↓
전처리 (ToTensor + Normalize)
  픽셀값 0~255 → -1.0~1.0
        ↓
DataLoader
  60,000개를 64개씩 배치로 분할
        ↓
모델 (순전파)
  Flatten → Linear → ReLU → Linear → ReLU → Linear
  784     →   128  →       →   64   →       →  10
        ↓
CrossEntropyLoss
  예측값과 정답의 차이 계산
        ↓
역전파 (loss.backward())
  각 weight의 기울기 계산
        ↓
Adam 옵티마이저 (optimizer.step())
  기울기로 weight 업데이트
        ↓
반복 (5 epochs × 937 배치)
        ↓
테스트 정확도: 95.82%
```

---

## 배운 내용

### 데이터 전처리

```python
transform = transforms.Compose([
    transforms.ToTensor(),           # PIL 이미지 → 텐서 (0~255 → 0.0~1.0)
    transforms.Normalize((0.5,), (0.5,))  # 정규화 → -1.0~1.0
])
```

- `ToTensor()` — 이미지를 PyTorch 텐서로 변환
- `Normalize()` — 픽셀값을 -1~1 범위로 조정해 학습 안정화

---

### DataLoader

```python
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
```

- 데이터를 배치 단위로 잘라서 모델에 공급
- `batch_size=64` — 한 번에 64개씩 학습
- `shuffle=True` — 매 epoch마다 순서 섞기 (학습 데이터만)

---

### 모델 구조

```
입력: 28×28 이미지 → Flatten → 784개 값
        ↓
Linear(784 → 128) + ReLU
        ↓
Linear(128 → 64) + ReLU
        ↓
Linear(64 → 10)  ← 0~9 클래스
```

**파라미터 수 계산:**

| 레이어 | weight | bias | 합계 |
|--------|--------|------|------|
| Linear(784→128) | 784×128 = 100,352 | 128 | 100,480 |
| Linear(128→64) | 128×64 = 8,192 | 64 | 8,256 |
| Linear(64→10) | 64×10 = 640 | 10 | 650 |
| **전체** | | | **109,386** |

---

### ReLU란?

```
ReLU(x) = max(0, x)

음수 → 0
양수 → 그대로
```

비선형성을 추가해서 모델이 복잡한 패턴을 학습할 수 있게 함.  
ReLU 없이 Linear만 쌓으면 아무리 깊어도 선형 모델과 같음.

---

### 역전파 (Backpropagation)

"Loss를 줄이려면 각 weight를 얼마나 바꿔야 하는지 계산하는 과정"

```
순전파: 입력 → layer1 → layer2 → 출력 → Loss
역전파: Loss → layer2 기울기 → layer1 기울기 (거꾸로)
```

```python
loss.backward()   # 모든 weight의 기울기 자동 계산
optimizer.step()  # 기울기로 weight 업데이트
```

---

### 학습 루프 구조

```python
for epoch in range(epochs):
    for images, labels in train_loader:
        outputs = model(images)        # 1. 순전파
        loss = criterion(outputs, labels)  # 2. Loss 계산
        optimizer.zero_grad()          # 3. 기울기 초기화
        loss.backward()                # 4. 역전파
        optimizer.step()               # 5. 가중치 업데이트
```

`optimizer.zero_grad()`를 매번 해야 하는 이유:  
PyTorch는 기울기를 누적하기 때문에 초기화하지 않으면 이전 배치의 기울기가 더해짐.

---

### model.eval() vs model.train()

```python
model.train()  # 학습 모드 — Dropout 등 활성화
model.eval()   # 평가 모드 — Dropout 등 비활성화
```

평가할 때는 `torch.no_grad()`도 함께 사용해서 메모리 절약.

---

## 실습 결과

```
Epoch 1/5, Loss: 0.3866
Epoch 2/5, Loss: 0.1818
Epoch 3/5, Loss: 0.1318
Epoch 4/5, Loss: 0.1068
Epoch 5/5, Loss: 0.0892

테스트 정확도: 95.82%
맞힌 개수: 9582 / 10000
```

한 번도 본 적 없는 테스트 데이터에서 95.82% 달성.  
틀린 이미지를 확인해보면 사람도 헷갈릴 만한 필체들이 대부분.

---

## 개념 참고

- 손실함수: [../00_concepts/loss_functions.md](../00_concepts/loss_functions.md)
- 옵티마이저: [../00_concepts/optimizers.md](../00_concepts/optimizers.md)

## 다음 단계

[03_expert_data](../03_expert_data/README.md) — CartPole 전문가 데이터 수집
