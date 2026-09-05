# Restauro di immagini degradate: confronto tra CNN, GAN e Transformer

Progetto per il corso di Deep Learning. L'obiettivo e' confrontare tre famiglie
di architetture sullo stesso problema: recuperare immagini che hanno subito una
degradazione realistica (sfocatura, sottocampionamento, rumore e compressione
JPEG applicati in cascata).

I tre modelli messi a confronto sono:

- **NAFNet / NAFSSR** (CNN), usato come baseline per denoising e super-resolution;
- **SRGAN / SRResNet** (GAN), addestrato sulle coppie degradate a risoluzione ridotta;
- **Uformer** (Transformer), addestrato per il denoising a piena risoluzione.

Il dataset di riferimento e' DIV2K. Le immagini degradate non sono scaricate:
vengono generate a partire dalle originali con `src/datasets/build_degraded_dataset.py`,
usando un seed fisso perche' i tre modelli lavorino esattamente sugli stessi dati.

## Struttura del repository

```
data/                    dataset e pesi pre-addestrati (non versionati)
docs/                    relazione, log dei run, documentazione dei progetti originali
experiments/configs/     configurazioni YAML di training e test
experiments/logs/        output dei run (non versionati)
figures/                 immagini usate nella documentazione
licenses/                licenze del codice di terze parti
notebooks/               esplorazione dei dati e visualizzazioni rapide
scripts/                 download dei dataset e dei pesi, entrypoint di training e test
src/datasets/            generazione del dataset degradato e dataloader
src/models/              architetture (SRGAN, Uformer, NAFNet)
src/training/            loop di training, loss, scheduler
src/evaluation/          test, inferenza, calcolo delle metriche
src/utils/               funzioni ausiliarie condivise
```

## Preparazione dell'ambiente

```bash
conda env create -f environment.yml
conda activate image-restoration
```

NAFNet usa il framework BasicSR, che va installato come pacchetto:

```bash
pip install -e src/models/nafnet
```

## Preparazione dei dati

Scaricare DIV2K in `data/` e generare le due versioni degradate, una a piena
risoluzione per il denoising e una a risoluzione ridotta per la super-resolution:

```bash
python -m src.datasets.build_degraded_dataset \
    --input data/DIV2K_train_HR --output data/DIV2K_train_degraded

python -m src.datasets.build_degraded_dataset \
    --input data/DIV2K_train_HR --output data/DIV2K_train_LR_noisy --keep-low-res
```

## Training

```bash
# Uformer
./scripts/train_uformer_denoise.sh

# SRResNet, poi SRGAN a partire dal generatore pre-addestrato
python -m src.training.train_srresnet --config_path experiments/configs/srgan/SRResNet_DIV2K_Noisy_128px.yaml
python -m src.training.train_srgan --config_path experiments/configs/srgan/SRGAN_DIV2K_Noisy_128px.yaml
```

## Valutazione

```bash
# Uformer, immagini intere
./scripts/test_uformer.sh <percorso_del_checkpoint>

# SRGAN
python -m src.evaluation.test_srgan --config_path experiments/configs/srgan/SRGAN_DIV2k_Test_128px.yaml

# baseline NAFNet
python -m src.evaluation.nafnet_denoise
python -m src.evaluation.nafnet_super_resolution
```

## Risultati

I log completi dei run sono in `docs/results/`, la discussione dei risultati
nella relazione (`docs/relazione.pdf`). In sintesi, sulle 100 immagini di
validazione DIV2K degradate:

| Modello | PSNR | SSIM |
| --- | --- | --- |
| Uformer, pesi originali (SIDD) | 23.67 dB | 0.576 |
| Uformer, fine-tuning 50 epoche | 25.34 dB | 0.713 |
| Uformer, fine-tuning 100 epoche | 25.41 dB | 0.714 |

## Codice di terze parti

Il repository include codice adattato da NAFNet, SRGAN-PyTorch e Uformer. La
corrispondenza tra file e progetto di origine, con le relative licenze, e' in
`licenses/README.md`.

## Autori

- Francesco Granata — Uformer
- Santi Lisi — NAFNet
- Dario — SRGAN
