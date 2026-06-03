# 04. Behavior Cloning (BC)

---

## Behavior Cloning이란?

전문가의 (state, action) 쌍을 지도 학습으로 모방하는 가장 단순한 Imitation Learning 방법.

```
전문가 데이터 (state, action) 10,000개
        ↓
지도 학습
"이 state에서 전문가는 이 action을 했다"
        ↓
모델이 전문가처럼 행동
```

MNIST에서 이미지 → 숫자를 학습한 것처럼,
BC는 state → action을 학습하는 것.

---

## 전체 흐름

```
전문가 데이터 수집 (03단계와 동일)
        ↓
numpy → PyTorch 텐서 변환
        ↓
TensorDataset + DataLoader 구성
        ↓
BCPolicy 모델 정의
        ↓
학습 루프 (10 epochs)
        ↓
성능 평가 (훈련 정확도 + 실제 환경 테스트)
```

---

## 모델 구조

```
입력: 관찰값 4개 (카트 위치, 카트 속도, 막대 각도, 막대 각속도)
        ↓
Linear(4→64) + ReLU
        ↓
Linear(64→64) + ReLU
        ↓
Linear(64→2)
        ↓
출력: 행동 2개 (0: 왼쪽, 1: 오른쪽)에 대한 점수
```

**파라미터 수 계산:**

| 레이어 | weight | bias | 합계 |
|--------|--------|------|------|
| Linear(4→64) | 4×64 = 256 | 64 | 320 |
| Linear(64→64) | 64×64 = 4,096 | 64 | 4,160 |
| Linear(64→2) | 64×2 = 128 | 2 | 130 |
| **전체** | | | **4,610** |

MNIST(109,386개)보다 훨씬 적은 이유: 입력이 784 → 4로 줄었기 때문.

---

## 학습 설정

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs    = 10
```

---

## 학습 결과

```
Epoch  1/10, Loss: 0.4702
Epoch  2/10, Loss: 0.3496
Epoch  3/10, Loss: 0.3383
Epoch  4/10, Loss: 0.3285
Epoch  5/10, Loss: 0.3143
Epoch  6/10, Loss: 0.2942
Epoch  7/10, Loss: 0.2679
Epoch  8/10, Loss: 0.2378
Epoch  9/10, Loss: 0.2071
Epoch 10/10, Loss: 0.1781
```

---

## 성능 평가 결과

```
훈련 데이터 정확도: 92.28% (9,228 / 10,000)
실제 환경 평균 스텝: 200.0 (10 에피소드)
```

**에이전트별 비교:**

| | 랜덤 에이전트 | BC 모델 | 전문가 |
|---|---|---|---|
| 평균 스텝 | 10~44 | 200.0 | 200.0 |
| 학습 방식 | 없음 | 모방 학습 | 규칙 기반 |

전문가 데이터 10,000개를 모방 학습한 것만으로 전문가와 동일한 성능 달성.

---

## MNIST와의 비교

| | MNIST | Behavior Cloning |
|---|---|---|
| 입력 | 이미지 (784 픽셀) | 상태 (4개 관찰값) |
| 정답 | 숫자 (0~9) | 행동 (0 또는 1) |
| 데이터 수 | 60,000개 | 10,000개 |
| 파라미터 수 | 109,386개 | 4,610개 |
| 최종 정확도 | 95.82% | 92.28% |

구조는 동일. 입력과 정답의 형태만 다를 뿐.

---

## 다음 단계

200 스텝 기준에서는 BC와 전문가가 동일한 성능을 보였음.  
하지만 이것만으로는 BC의 실제 한계를 확인하기 어려움.  
더 긴 스텝, 극단적인 초기 상태 등 다양한 조건에서 테스트가 필요함.

다음 단계: [../05_behavior_cloning_limits](../05_behavior_cloning_limits/README.md)

---

## 개념 참고

- Imitation Learning: [../00_concepts/imitation_learning.md](../00_concepts/imitation_learning.md)
- 파라미터: [../00_concepts/parameters.md](../00_concepts/parameters.md)
- 활성화 함수: [../00_concepts/activation_functions.md](../00_concepts/activation_functions.md)
- 손실함수: [../00_concepts/loss_functions.md](../00_concepts/loss_functions.md)
- NumPy vs PyTorch: [../00_concepts/numpy_vs_pytorch.md](../00_concepts/numpy_vs_pytorch.md)
