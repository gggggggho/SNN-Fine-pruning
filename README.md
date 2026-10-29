# SNN Fine-Pruning

[HS-FP](https://doi.org/10.3390/electronics15050937)와 SS-FP를 이용해 뉴로모픽 이벤트 데이터로 학습한 스파이킹 신경망(SNN)의 백도어를 제거하는 연구 코드입니다. 공격 모델 학습 코드는 `Sneaky-Spikes-main`의 static, moving, smart 트리거 구현을 참고합니다.

## 방법

- **HS-FP**: 깨끗한 학습 데이터에 대한 은닉층의 평균 활성값을 hook으로 측정하고, 활성값이 낮은 완전연결층 뉴런부터 순차적으로 제거합니다. 이후 surrogate gradient로 미세 조정합니다.
- **SS-FP**: 깨끗한 데이터에서 뉴런의 누적 spike 수로 제거 순서를 정합니다. 미세 조정 시 convolution 층은 STDP, 나머지 학습 파라미터는 surrogate gradient를 사용합니다.
- 두 방법은 static, moving, smart 백도어 공격과 N-MNIST, DVS128-Gesture 데이터셋에 대해 평가되었습니다.

## 저장소 구성

| 경로 | 용도 |
|---|---|
| `main.py` | static, moving, smart 트리거를 이용한 백도어 SNN 학습 |
| `dynamic.py` | dynamic 트리거 실험 코드 |
| `defense/base.py` | 방어 모듈이 공유하는 기본 클래스 |
| `defense/fp.py` | HS-FP 방어 구현 |
| `defense/ssfp.py` | SS-FP 방어 구현 |
| `models.py`, `datasets.py`, `poisoned_dataset.py` | SNN 모델, 데이터셋 로딩, 트리거 데이터 생성 |

`dynamic.py`의 dynamic 공격은 참고 코드에 포함되어 있으며, HS-FP/SS-FP 평가에 사용된 세 가지 공격에는 포함되지 않습니다.

## 환경 설치

Python 3.12, PyTorch 2.6.0, SpikingJelly 0.0.0.0.14 환경에서 수행되었습니다. 저장소의 패키지 목록을 설치하려면 다음을 실행합니다.

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

## 백도어 모델 학습

아래 명령은 N-MNIST에서 기본 공격 설정(오염률 10%, 트리거 크기 10%, 타깃 라벨 1, 극성 1, seed 42)으로 static, moving, smart 모델을 각각 학습합니다. `main.py --type`의 허용값은 `static`, `moving`, `smart`입니다.

```bash
python main.py --dataset mnist --T 16 --type static --trigger_label 1 --polarity 1 --pos top-left --trigger_size 0.1 --epsilon 0.1 --batch_size 16 --epochs 10 --seed 42
python main.py --dataset mnist --T 16 --type moving --trigger_label 1 --polarity 1 --pos top-left --trigger_size 0.1 --epsilon 0.1 --batch_size 16 --epochs 10 --seed 42
python main.py --dataset mnist --T 16 --type smart --trigger_label 1 --polarity 1 --trigger_size 0.1 --epsilon 0.1 --batch_size 16 --epochs 10 --seed 42 --least --n_masks 2
```

생성되는 체크포인트 경로는 다음과 같습니다.

| 트리거 | 체크포인트 |
|---|---|
| static | `experiments/mnist_static_0.1_0.1_top-left_1_42/model.pth` |
| moving | `experiments/mnist_moving_0.1_0.1_top-left_1_42/model.pth` |
| smart | `experiments/mnist_smart_0.1_0.1_42/model.pth` |

## HS-FP 적용

앞 절에서 학습한 각 모델에 HS-FP를 적용합니다. 방어 스크립트의 데이터셋 이름은 `nmnist`이며, 스마트 트리거 모델에는 학습 때와 같이 `--least`를 지정합니다.

```bash
python defense/fp.py --dataset nmnist --model_path experiments/mnist_static_0.1_0.1_top-left_1_42/model.pth --T 16 --trigger_type static --trigger_label 1 --polarity 1 --pos top-left --trigger_size 0.1 --epsilon 0.1 --once_prune_ratio 0.05 --acc_ratio 0.8 --batch_size 32 --epochs 20 --device cuda --random_seed 42 --frequency_save 0 --non_blocking true
python defense/fp.py --dataset nmnist --model_path experiments/mnist_moving_0.1_0.1_top-left_1_42/model.pth --T 16 --trigger_type moving --trigger_label 1 --polarity 1 --pos top-left --trigger_size 0.1 --epsilon 0.1 --once_prune_ratio 0.05 --acc_ratio 0.8 --batch_size 32 --epochs 20 --device cuda --random_seed 42 --frequency_save 0 --non_blocking true
python defense/fp.py --dataset nmnist --model_path experiments/mnist_smart_0.1_0.1_42/model.pth --T 16 --trigger_type smart --trigger_label 1 --polarity 1 --trigger_size 0.1 --epsilon 0.1 --least --n_masks 2 --once_prune_ratio 0.05 --acc_ratio 0.8 --batch_size 32 --epochs 20 --device cuda --random_seed 42 --frequency_save 0 --non_blocking true
```

## SS-FP 적용

같은 세 백도어 모델에 SS-FP를 적용합니다. 옵션은 HS-FP와 같고, 실행 파일만 `defense/ssfp.py`로 바꿉니다.

```bash
python defense/ssfp.py --dataset nmnist --model_path experiments/mnist_static_0.1_0.1_top-left_1_42/model.pth --T 16 --trigger_type static --trigger_label 1 --polarity 1 --pos top-left --trigger_size 0.1 --epsilon 0.1 --once_prune_ratio 0.05 --acc_ratio 0.8 --batch_size 32 --epochs 20 --device cuda --random_seed 42 --frequency_save 0 --non_blocking true
python defense/ssfp.py --dataset nmnist --model_path experiments/mnist_moving_0.1_0.1_top-left_1_42/model.pth --T 16 --trigger_type moving --trigger_label 1 --polarity 1 --pos top-left --trigger_size 0.1 --epsilon 0.1 --once_prune_ratio 0.05 --acc_ratio 0.8 --batch_size 32 --epochs 20 --device cuda --random_seed 42 --frequency_save 0 --non_blocking true
python defense/ssfp.py --dataset nmnist --model_path experiments/mnist_smart_0.1_0.1_42/model.pth --T 16 --trigger_type smart --trigger_label 1 --polarity 1 --trigger_size 0.1 --epsilon 0.1 --least --n_masks 2 --once_prune_ratio 0.05 --acc_ratio 0.8 --batch_size 32 --epochs 20 --device cuda --random_seed 42 --frequency_save 0 --non_blocking true
```

모든 방어 결과는 기본적으로 `experiments/` 아래에 저장됩니다. DVS128-Gesture 모델을 방어할 때는 체크포인트와 `--dataset gesture`를 사용하고, `--acc_ratio 0.3 --epochs 128`로 바꿉니다. GPU가 없으면 `--device cpu`를 지정합니다. `--save_path`를 학습 명령에 지정했다면 방어 명령의 `--model_path`도 해당 폴더로 바꿉니다.

## 실험 결과

아래는 fine-tuning 종료 시 보고한 정확도(ACC)와 공격 성공률(ASR) 범위입니다. 각 범위는 static, moving, smart 공격 결과를 포함합니다.

| 방법 | 데이터셋 | ACC | ASR |
|---|---|---:|---:|
| HS-FP | N-MNIST | 99.25–99.34% | 9.94–12.35% |
| HS-FP | DVS128-Gesture | 67.01–75.69% | 0.39–7.64% |
| SS-FP | N-MNIST | 96.02–96.50% | 10.02–10.24% |
| SS-FP | DVS128-Gesture | 67.01–72.22% | 6.94–11.11% |

