# Vision Trainer

Application locale pour entraîner des modèles **YOLO / Ultralytics** :

- **Détection** — objets + bounding boxes
- **Classification** — classe / probabilités (ImageFolder, style Teachable Machine)

## Installation (local)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Lancement (local / CPU)

```bash
streamlit run app/streamlit_app.py
```

Données locales par défaut : `./artifacts/` (`datasets/`, `runs/`).

## Datasets

### Détection (YOLO)

ZIP avec `data.yaml`, **ou** sans YAML mais avec :

```text
train/images + train/labels (+ val/…)
train/classes.txt   # une classe par ligne
```

Dans ce second cas, VisionTrainer génère un `data.yaml` interne (sans écraser un YAML existant).

### Classification

```text
dataset/
  train/
    ClasseA/
    ClasseB/
  val/
    ClasseA/
    ClasseB/
```

Import ZIP sur **Classification — Dataset**. Aucune bounding box.

## Inférence classification — seuil « Inconnu »

Sur la page Inférence (mode Classification) :

- **seuil de confiance minimum** → sous le seuil : `INCONNU / CONFIANCE INSUFFISANTE`
- **écart Top-1 / Top-2** → trop faible : `INCERTAIN`

Ce n’est **pas** une détection OOD formelle.

## Lancement avec Docker

Prérequis : Docker et Docker Compose.

```bash
docker compose up --build
```

Puis : http://localhost:8501

Upload Streamlit : `maxUploadSize = 2048` (Mo) dans `.streamlit/config.toml`.

### Données persistantes

Bind mount `./data` → `/app/data` :

- `datasets/` — ZIP importés
- `runs/` — entraînements / `best.pt`
- `ultralytics/` — cache
- `tmp/` — temporaires

`VISIONTRAINER_DATA_DIR=/app/data` (Docker). Hors Docker : `./artifacts`.

### Docker Desktop + WSL

```bash
docker compose -f docker-compose.yml -f docker-compose.wsl.yml up --build
```

## Docker GPU NVIDIA

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Vérifier CUDA :

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml exec vision-trainer \
  python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
```

## Tests

```bash
pytest
```

## Artefacts

Runs sous `<data_root>/runs/<run_id>/` avec `task=detect|classify` dans `status.json` (les anciens runs sans `task` restent lus comme détection).
