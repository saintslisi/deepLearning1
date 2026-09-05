# Dati

Questa cartella non e' versionata: contiene i dataset e i pesi pre-addestrati,
che vanno scaricati o generati in locale.

Struttura attesa dagli script e dalle configurazioni:

```
data/
  DIV2K_train_HR/          originali ad alta risoluzione
  DIV2K_valid_HR/
  DIV2K_train_degraded/    generate con build_degraded_dataset.py
  DIV2K_valid_degraded/
  DIV2K_train_LR_noisy/    versione a risoluzione ridotta, per SRGAN
  DIV2K_valid_LR_noisy/
  pretrained_models/       pesi scaricati con scripts/download_weights.sh
```
