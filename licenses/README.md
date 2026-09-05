# Licenze del codice di terze parti

Parte del codice in `src/` deriva da progetti open source e mantiene la licenza
originale. Qui sotto la corrispondenza tra i file di questo repository e il
progetto da cui provengono.

| Origine | Licenza | File derivati |
| --- | --- | --- |
| [NAFNet](https://github.com/megvii-research/NAFNet) (MEGVII Technology) | `LICENSE-NAFNet` | `src/models/nafnet/`, `src/datasets/prepare_*.py`, `src/datasets/make_pickle.py`, `src/evaluation/nafnet_*.py`, `experiments/configs/nafnet/` |
| [SRGAN-PyTorch](https://github.com/Lornatang/SRGAN-PyTorch) (Lornatang) | `LICENSE-SRGAN-Lornatang` | `src/models/srgan.py`, `src/datasets/srgan_dataset.py`, `src/training/train_srgan.py`, `src/training/train_srresnet.py`, `src/evaluation/test_srgan.py`, `src/evaluation/inference_srgan.py`, `src/evaluation/image_quality_assessment.py`, `src/utils/imgproc.py`, `src/utils/srgan_utils.py` |
| [Uformer](https://github.com/ZhendongWang6/Uformer) (Zhendong Wang et al., MIT) | vedi repository originale | `src/models/uformer.py`, `src/datasets/uformer_*.py`, `src/training/train_uformer*.py`, `src/training/losses.py`, `src/training/options.py`, `src/training/warmup_scheduler/`, `src/evaluation/test_uformer_*.py`, `src/utils/` (image, model, dir, dataset utils) |

Il codice scritto da noi (pipeline di degradazione, script di confronto, baseline
NAFNet su DIV2K, adattamenti per MPS) non e' coperto da queste licenze.
