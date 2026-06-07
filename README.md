# PyTorch Behavior Cloning Study

PyTorch 기초부터 Behavior Cloning 구현까지, Robot AI 연구의 기반을 직접 쌓아가는 실습 레포.

---

## 공부 순서

| 단계 | 주제 | 상태 |
|------|------|------|
| 00 | 개념 정리 | 🔄 계속 추가 중 |
| 01 | 텐서 기초 + 자동미분 + 선형 회귀 | ✅ 완료 |
| 02 | 신경망 구현 (MNIST 분류기) | ✅ 완료 |
| 03 | CartPole 전문가 데이터 수집 | ✅ 완료 |
| 04 | Behavior Cloning 학습 및 평가 | ✅ 완료 |
| 05 | BC의 한계 탐구 (Distribution Shift 직접 확인) | ✅ 완료 |

---

## 00_concepts 폴더에 대해

공부를 진행하면서 생긴 질문이나 핵심 기술 개념을 정리하는 폴더.
정해진 순서 없이 궁금한 게 생길 때마다 추가함.

```
예: "ReLU는 왜 쓰는 거지?" → activation_functions.md 추가
    "배치가 뭐야?" → batch.md 추가
    "NumPy랑 PyTorch는 뭐가 달라?" → numpy_vs_pytorch.md 추가
```

각 단계를 진행하면서 자연스럽게 쌓이는 개인 레퍼런스.

---

## 05단계를 추가한 이유

원래 이 레포는 01~04단계까지만 계획되어 있었음.

04단계에서 BC 모델이 200 스텝 기준으로 전문가와 동일한 성능을 보였음.

그런데 자연스럽게 이런 의문이 생겼음:

> "200 스텝이 CartPole의 기본 최대값인데, 이게 진짜 잘하는 건지 그냥 버티는 건지 어떻게 알지?"
> "더 어려운 상황에서도 잘 동작할까?"

이 질문들을 직접 확인해보고 싶어서 05단계를 추가함.
정해진 커리큘럼이 아니라 공부하면서 생긴 궁금증을 스스로 탐구하는 과정.

**05단계에서 확인한 것들:**
- 최대 스텝 1000으로 늘렸을 때 BC vs 전문가 차이
- 극단적인 초기 상태에서 Distribution Shift 직접 확인
- 초기 각도별 성능 비교 그래프

→ BC는 학습 데이터 분포를 벗어나면 급격히 무너지는 한계를 직접 눈으로 확인함.
→ 이 한계가 다음 레포(pytorch-diffusion-study)로 이어지는 출발점.

---

## 목표

- PyTorch 기초 익히기
- Imitation Learning의 핵심인 Behavior Cloning 직접 구현
- BC의 한계를 직접 확인하고 다음 연구 방향 탐색
- Robot AI 연구(Diffusion Policy, Inverse Constraint Learning 등)의 기반 개념 이해

---

## 다음 레포

BC의 한계를 직접 확인했으니, 이를 해결하는 방향으로 이어짐.

```
Distribution Shift  → IRL 계열이 해결
Multimodal 문제     → Diffusion Policy가 해결
```

👉 [pytorch-diffusion-study](https://github.com/Rei5ende/pytorch-diffusion-study) (예정)

---

## 이 레포를 만든 이유

Behavior Cloning은 Robot AI 연구의 공통 기반임.

```
Behavior Cloning (모방 학습)
  → 전문가 시연 데이터로 정책 학습
        ↓
Diffusion Policy
  → 거친 시연 데이터로 다양한 행동 생성
        ↓
Inverse Constraint Learning
  → 시연에서 제약 조건을 역추론
```

직접 구현해보면서 최신 Robot AI 연구들이 해결하려는 문제를 더 깊이 이해하는 것이 목표.

---

## 참고

- 관련 논문 리뷰: https://github.com/Rei5ende/robotics-paper-study
