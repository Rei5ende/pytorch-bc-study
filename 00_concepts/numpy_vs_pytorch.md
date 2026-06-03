# NumPy vs PyTorch

> 둘 다 배열을 다루는 라이브러리지만 역할이 다름

---

## 핵심 차이 한 줄 요약

```
NumPy  = 데이터 처리 도구 (계산기)
PyTorch = 딥러닝 학습 도구 (자동미분 + GPU)
```

---

## NumPy

```python
import numpy as np
a = np.array([1.0, 2.0, 3.0])
```

- 수치 계산 전용 라이브러리
- CPU에서만 동작
- 자동미분 없음
- 데이터 처리, 전처리에 주로 사용

---

## PyTorch

```python
import torch
a = torch.tensor([1.0, 2.0, 3.0])
```

- 딥러닝 전용 라이브러리
- CPU + GPU 모두 동작
- 자동미분 지원 ← 핵심 차이
- 모델 학습에 사용

---

## 핵심 차이: 자동미분

```python
# NumPy → 기울기 계산 불가
a = np.array([3.0])
b = a ** 2
# b.backward() ← 존재하지 않음

# PyTorch → 기울기 자동 계산
a = torch.tensor([3.0], requires_grad=True)
b = a ** 2
b.backward()
print(a.grad)  # 6.0
```

자동미분이 없으면 역전파를 할 수 없어서 딥러닝 학습 불가능.

---

## 비교표

| | NumPy | PyTorch |
|---|---|---|
| 주 용도 | 데이터 처리 | 딥러닝 학습 |
| 자동미분 | ❌ | ✅ |
| GPU 지원 | ❌ | ✅ |
| 속도 | 빠름 (단순 연산) | 빠름 (GPU 사용 시) |
| 주요 타입 | `np.array` | `torch.tensor` |

---

## 왜 같이 쓰나?

gym 환경이 기본적으로 numpy 배열을 반환하기 때문에 전처리는 numpy로 하고, 학습 단계에서 PyTorch로 변환하는 방식을 주로 사용함.

```python
# 데이터 수집은 NumPy로 (gym이 numpy 반환)
states = []
states.append(obs)
states = np.array(states)

# 학습은 PyTorch로 (자동미분 필요)
states_tensor = torch.FloatTensor(states)
```

```
gym 환경 → numpy → (변환) → PyTorch 텐서 → 모델 학습
```

---

## FloatTensor vs LongTensor

PyTorch에서 데이터 타입에 따라 텐서 종류가 다름.

```python
torch.FloatTensor(states)   # 실수형 (state: 연속적인 값)
torch.LongTensor(actions)   # 정수형 (action: 0 또는 1)
```

CrossEntropyLoss는 정답 레이블로 정수형(Long)을 요구하기 때문에 action은 반드시 LongTensor로 변환해야 함.

---

## 참고

- 자동미분 개념: [../01_tensor_basics/README.md](../01_tensor_basics/README.md)
- 손실함수: [loss_functions.md](./loss_functions.md)
