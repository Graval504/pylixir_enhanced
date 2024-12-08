Pylixir-Enhanced
============
2024년 2학기 강화학습 프로젝트로, 기존 Pylixir 모델을 개선하여 성능을 높이는 것을 목표로 하는 프로젝트입니다.

Install
=========

해당 라이브러리는 의존성 관리를 위해 `poetry` 를 사용합니다. 
[poetry home](https://python-poetry.org/docs/) 에서 poetry를 설치해 주세요.

```bash
git clone https://github.com/oleneyl/pylixir
poetry install
```
해당 포크는 효율적인 학습을 위해 cuda 11.8 기반의 pytorch를 설치합니다.\
따라서 Non-Nvidia GPU 또는 Geforce 700 시리즈 이하의 제품군은 제대로 작동하지 않을 가능성이 있으므로\
이 경우 원본 repo의 pyproject.toml을 참고하실 것을 부탁드립니다.


Deep Learning
==============

현재 개선된 모델의 성능은 합14이상 49.4%, 합16이상 16.56%, 합18이상 2.92%으로, 기존 Pylixir 및 엘파고v1 모델보다 더 좋은 성능을 가지고 있습니다.
- Model Size는 약 24.6Mb입니다.

[Benchmark](benchmark.md)

Improvement
============
모델 및 코드 개선 사항입니다.
- Transformer Layer를 6으로 증가
- Transformer Layer의 활성화함수를 Swish(SiLU)로 변경
- output Layer를 Linear-ReLU-Linear 형태에서 [SwiGLU FFN](https://arxiv.org/pdf/2002.05202)기반으로 변경
- loss를 smooth-l1 loss에서 huber loss($\delta=2.5$)로 변경
- 1, 2번째 옵션이 아닌 아무 옵션이나 제일 높이도록 loss 계산을 변경
- Cosine annealing LR Scheduler 구현 및 사용
- 학습에 GPU를 사용하도록 변경

TODO
===========
추후 예정 혹은 기획중인 모델 및 코드 개선 사항입니다.
- Transformer Layer 수 증가
- 옵션 저격용 모델 학습
- GNN(Graph Neural Network) 기반 모델 학습
- evaluation 병렬화

Train with best configuration
===========

```sh
poetry run python deep/stable_baselines/train.py deep/conf/dqn_transformer.yaml
```

Evaluate
===========
```sh
poetry run python deep/stable_baselines/evaluate.py $ZIP_FILE_PATH
```


