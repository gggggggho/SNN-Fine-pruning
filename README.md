# SNN Fine-Pruning

[HS-FP](https://doi.org/10.3390/electronics15050937)와 SS-FP를 이용해 뉴로모픽 이벤트 데이터로 학습한 스파이킹 신경망(SNN)의 백도어를 제거하는 연구 코드입니다. 공격 모델 학습 코드는 `Sneaky-Spikes-main`의 static, moving, smart 트리거 구현을 참고합니다.

## 방법

- **HS-FP**: 깨끗한 학습 데이터에 대한 은닉층의 평균 활성값을 hook으로 측정하고, 활성값이 낮은 완전연결층 뉴런부터 순차적으로 제거합니다. 이후 surrogate gradient로 미세 조정합니다.
- **SS-FP**: 깨끗한 데이터에서 뉴런의 누적 spike 수로 제거 순서를 정합니다. 미세 조정 시 convolution 층은 STDP, 나머지 학습 파라미터는 surrogate gradient를 사용합니다.
- 두 방법은 논문의 static, moving, smart 백도어 공격과 N-MNIST, DVS128-Gesture 데이터셋에 대해 평가되었습니다.

## 저장소 구성

| 경로 | 용도 |
|---|---|
| `main.py` | static, moving, smart 트리거를 이용한 백도어 SNN 학습 |
| `dynamic.py` | dynamic 트리거 실험 코드 |
| `defense/base.py` | 방어 모듈이 공유하는 기본 클래스 |
| `defense/fp.py` | HS-FP 방어 구현 |
| `defense/ssfp.py` | SS-FP 방어 구현 |
| `models.py`, `datasets.py`, `poisoned_dataset.py` | SNN 모델, 데이터셋 로딩, 트리거 데이터 생성 |

`dynamic.py`의 dynamic 공격은 참고 코드에 포함되어 있으며, 첨부 논문의 HS-FP/SS-FP 평가에 사용된 세 가지 공격에는 포함되지 않습니다.

## 환경 설치

논문 실험은 Python 3.12, PyTorch 2.6.0, SpikingJelly 0.0.0.0.14 환경에서 수행되었습니다. 저장소의 패키지 목록을 설치하려면 다음을 실행합니다.

```bash
python -m pip install -r requirements.txt
```

GPU 가속은 CUDA와 호환되는 PyTorch 및 CuPy 환경이 필요합니다. 학습 명령에 `--cupy`를 추가하면 SpikingJelly의 CuPy backend를 사용합니다.

## 데이터 준비

데이터 루트는 기본적으로 `data/`이며, 데이터셋별 하위 폴더를 사용합니다.

```text
data/
├── mnist/    # N-MNIST 원본 아카이브 또는 SpikingJelly가 인식하는 데이터
└── gesture/  # DVS128-Gesture 원본 아카이브 또는 추출 데이터
```

- [N-MNIST](https://www.garrickorchard.com/datasets/n-mnist)는 수동으로 내려받아 `data/mnist/`에 둡니다.
- [DVS128-Gesture](https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794)는 내려받아 `data/gesture/`에 둡니다.
- 방어 코드에서 N-MNIST를 지정할 때 데이터셋 이름은 `nmnist`입니다. `defense/fp.py`와 `defense/ssfp.py`는 `data/mnist/`만 있고 `data/nmnist/`가 없으면 별칭 폴더를 준비합니다.

## 백도어 모델 학습 예시

N-MNIST에 static 트리거를 넣어 10 epoch 동안 학습하는 예시입니다. 논문 설정에 맞춰 오염률과 트리거 크기를 10%, 타깃 라벨과 극성을 1로 지정했습니다.

```bash
python main.py --dataset mnist --T 16 --type static --trigger_label 1 --polarity 1 --pos top-left --trigger_size 0.1 --epsilon 0.1 --batch_size 16 --epochs 10
```

`--type`은 `static`, `moving`, `smart` 중에서 선택합니다. smart 공격에서 비활성 영역을 선택하려면 `--least`를 추가합니다. `--data_dir`와 `--save_path`로 데이터 및 결과 위치를 바꿀 수 있습니다. 학습된 모델은 기본적으로 `experiments/<실험 설정>/model.pth`에 저장됩니다.

## 논문 결과

아래는 논문에서 fine-tuning 종료 시 보고한 정확도(ACC)와 공격 성공률(ASR) 범위입니다. 각 범위는 static, moving, smart 공격 결과를 포함합니다.

| 방법 | 데이터셋 | ACC | ASR |
|---|---|---:|---:|
| HS-FP | N-MNIST | 99.25–99.34% | 9.94–12.35% |
| HS-FP | DVS128-Gesture | 67.01–75.69% | 0.39–7.64% |
| SS-FP | N-MNIST | 96.02–96.50% | 10.02–10.24% |
| SS-FP | DVS128-Gesture | 67.01–72.22% | 6.94–11.11% |

이는 논문에 기재된 결과이며, 이 저장소에서 동일한 수치가 재현되었음을 의미하지 않습니다. 논문은 두 방법 모두 완전연결층 뉴런의 90% 이상을 제거해야 방어 효과를 얻는 경우가 있었고, DVS128-Gesture에서는 제한된 데이터와 높은 시공간 복잡도로 정확도 회복이 불안정할 수 있다고 보고합니다.

## 방어 실행 예시

백도어 학습 모델과 데이터가 준비되어 있다면 N-MNIST HS-FP를 다음과 같이 실행합니다. 논문 설정인 pruning 비율 0.05, 허용 정확도 비율 0.8, fine-tuning 20 epoch를 사용합니다.

```bash
python defense/fp.py --dataset nmnist --model_path experiments/mnist_static_0.1_0.1_top-left_1_42/model.pth --T 16 --trigger_type static --trigger_label 1 --polarity 1 --pos top-left --trigger_size 0.1 --epsilon 0.1 --once_prune_ratio 0.05 --acc_ratio 0.8 --batch_size 32 --epochs 20 --device cuda --random_seed 42 --frequency_save 0 --non_blocking true
```

SS-FP는 같은 인자에서 `defense/fp.py`를 `defense/ssfp.py`로 바꿔 실행합니다. DVS128-Gesture는 `--dataset gesture`, Gesture 모델 경로, `--acc_ratio 0.3`, `--epochs 128`을 사용합니다. GPU가 없으면 `--device cpu`로 바꿀 수 있습니다.

