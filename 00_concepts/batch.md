# Batch (배치)

> 전체 데이터를 한 번에 학습하지 않고 작은 묶음으로 나눠서 학습하는 단위

---

## 왜 나눠서 학습하나?

```
전체 데이터 10,000개를 한 번에 학습하면?

❌ 메모리 부족
   10,000개를 동시에 메모리에 올리면 터질 수 있음

❌ 학습이 불안정
   한 번에 너무 많은 데이터를 보면 평균값으로 수렴해버림

❌ weight 업데이트 횟수 부족
   한 번 업데이트하려고 10,000개를 다 계산해야 함
```

---

## 배치로 나누면?

```
전체 데이터: 10,000개
배치 크기:       64개
배치 수:        157개 (10,000 ÷ 64)

1배치(64개) 학습 → weight 업데이트
2배치(64개) 학습 → weight 업데이트
...
157배치까지 반복 → 1 epoch 완료
```

weight 업데이트가 157번 일어남.  
전체를 한 번에 하면 1번밖에 안 되는 것과 비교하면 훨씬 많이 학습됨.

---

## 직관적인 비유

```
교과서 1000페이지를 공부할 때:

한 번에 다 읽고 복습 (배치 없음)
→ 너무 많아서 기억 못 함

50페이지씩 나눠서 읽고 복습 (배치 있음)
→ 조금씩 반복하면서 더 잘 기억
```

---

## 코드에서 보면

```python
DataLoader(dataset, batch_size=64, shuffle=True)

for states_batch, actions_batch in dataloader:
    # states_batch.shape: (64, 4) ← 64개씩 잘라서 들어옴
    # actions_batch.shape: (64,)

    pred = model(states_batch)
    loss = criterion(pred, actions_batch)
    loss.backward()
    optimizer.step()
    # ↑ 64개마다 한 번씩 weight 업데이트
```

---

## 관련 용어

| 용어 | 의미 |
|------|------|
| batch_size | 한 번에 학습하는 데이터 수 |
| epoch | 전체 데이터를 한 바퀴 다 학습한 것 |
| iteration | 배치 하나를 학습한 것 (= weight 업데이트 1회) |

```
1 epoch = 배치 수(157)번의 iteration
5 epochs = 157 × 5 = 785번의 weight 업데이트
```

---

## 프로젝트별 배치 수 비교

| | MNIST | Behavior Cloning |
|---|---|---|
| 전체 데이터 | 60,000개 | 10,000개 |
| 배치 크기 | 64 | 64 |
| 배치 수 | 937 | 157 |
| epoch당 업데이트 횟수 | 937번 | 157번 |

---

## 참고

- DataLoader 사용: [../02_neural_network/README.md](../02_neural_network/README.md)
- 옵티마이저: [optimizers.md](./optimizers.md)
