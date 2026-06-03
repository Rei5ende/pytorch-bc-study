# Imitation Learning (모방 학습)

> 전문가의 행동을 보고 따라 하도록 학습하는 방법

---

## 왜 필요한가 — 강화학습의 한계

로봇을 학습시키는 가장 직관적인 방법은 강화학습(RL)임.

```
강화학습:
  잘 하면 보상 +10
  못 하면 패널티 -5
  → 반복하면서 스스로 학습
```

하지만 현실에서는 문제가 있음:

```
❌ 보상 설계가 어려움
   "로봇이 컵을 안전하게 옮기는 것"을 숫자로 어떻게 표현?

❌ 학습 시간이 매우 오래 걸림
   처음엔 완전 랜덤 → 수백만 번 시도 필요

❌ 실제 로봇에서는 위험
   학습 중 실패 = 로봇 파손 또는 사람 부상
```

---

## Imitation Learning의 핵심 아이디어

> "전문가가 하는 걸 보여주면 되잖아?"

```
강화학습:  보상 신호로 스스로 학습 (시행착오)
모방 학습: 전문가 시연을 보고 따라 학습 (선생님 필요)
```

---

## 대표적인 방법 2가지

### 1. Behavior Cloning (BC)

가장 단순한 모방 학습.

```
전문가 데이터: (state, action) 쌍 수집
        ↓
지도 학습으로 학습
"이 state에서 전문가는 이 action을 했다"
        ↓
모델이 전문가처럼 행동
```

MNIST에서 이미지 → 숫자를 학습한 것처럼
BC는 state → action을 학습하는 것.

**장점:** 구현이 간단, 학습이 빠름
**단점:** Distribution Shift 문제 발생 가능

```
Distribution Shift:
  학습: 전문가 데이터만 봄
  실제: 처음 보는 상황 발생
  → 한 번 실수하면 계속 낯선 상황 → 실패 반복
```

---

### 2. Inverse Reinforcement Learning (IRL)

전문가 행동에서 보상 함수를 역으로 추론.

```
전문가 시연
        ↓
"전문가가 이렇게 행동한 이유는 뭘까?"
        ↓
보상 함수 역추론
        ↓
추론된 보상으로 RL 학습
```

**장점:** 새로운 환경에도 일반화 가능
**단점:** 구현이 복잡, 계산 비용 높음

---

## RIRO Lab 연구와의 연결

```
Behavior Cloning
  → 전문가 시연에서 행동 직접 학습
        ↓
Inverse Constraint Learning (TCL, ILCL)
  → 시연에서 행동이 아닌 제약 조건을 역추론
  → 새로운 상황에도 제약을 일반화

DiSPo
  → BC의 한계(Temporal Structure 부재) 해결
  → 거친 시연 데이터로 정밀한 행동 생성
```

---

## 이 레포에서의 흐름

```
03_expert_data
  전문가 데이터 수집 (state, action 쌍)
        ↓
04_behavior_cloning
  BC 모델 학습 및 평가
```

---

## 참고

- Behavior Cloning 구현: [../04_behavior_cloning/README.md](../04_behavior_cloning/README.md)
- 손실함수: [loss_functions.md](./loss_functions.md)
- 옵티마이저: [optimizers.md](./optimizers.md)
