# PyTorch Behavior Cloning Study

RIRO Lab 인턴십 준비를 위한 PyTorch 기초 및 Behavior Cloning 구현 실습 레포입니다.

---

## 공부 순서

| 단계 | 주제 | 상태 |
|------|------|------|
| 01 | 텐서 기초 + 자동미분 + 선형 회귀 | ✅ 완료 |
| 02 | 신경망 구현 (MNIST 분류기) | ✅ 완료 |
| 03 | CartPole 전문가 데이터 수집 | ✅ 완료 |
| 04 | Behavior Cloning 학습 및 평가 | ✅ 완료 |
| 05 | BC의 한계 탐구 | 🔄 진행 중 |

---

## 05단계를 추가한 이유

04단계에서 BC 모델이 200 스텝 기준으로 전문가와 동일한 성능을 보였음.

그런데 자연스럽게 이런 의문이 생겼음:

> "200 스텝이 CartPole의 기본 최대값인데, 이게 진짜 잘하는 건지 그냥 버티는 건지 어떻게 알지?"
> "더 어려운 상황에서도 잘 동작할까?"

이 질문들을 직접 확인해보고 싶어서 05단계를 추가함.  
정해진 커리큘럼이 아니라 공부하면서 생긴 궁금증을 스스로 탐구하는 과정.

---

## 목표

- PyTorch 기초 익히기
- Imitation Learning의 핵심인 Behavior Cloning 직접 구현
- RIRO Lab의 DiSPo, ICL 연구의 기반 개념 이해
- BC의 한계를 직접 확인하고 다음 연구 방향 탐색

---

## 이 레포를 만든 이유

Behavior Cloning은 RIRO Lab의 핵심 연구인 DiSPo, Inverse Constraint Learning의 공통 기반임.

```
Behavior Cloning (모방 학습)
  → 전문가 시연 데이터로 정책 학습
        ↓
DiSPo
  → 거친 시연 데이터로 정밀한 동작 생성
        ↓
Inverse Constraint Learning (TCL, ILCL)
  → 시연에서 제약 조건을 역추론
```

직접 구현해보면서 논문이 해결하려는 문제를 더 깊이 이해하는 것이 목표.

---

## 참고

- RIRO Lab: https://rirolab.kaist.ac.kr
- 관련 논문 리뷰: https://github.com/Rei5ende/riro-paper-study
