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

## Pipelines

Chaînage optionnel **Détection → Classification** (classe par classe) :

1. Entraîner un détecteur (ex. `Apple`, `Tomato`, `Carrot`)
2. Entraîner un ou plusieurs classificateurs (ex. Golden / Gala / Granny Smith)
3. Page **Pipelines** : choisir le détecteur, cocher les classes à affiner, associer un classificateur + seuils
4. Page **Inférence Pipeline** : lancer une image

Fonctionnement :

- détection sur l’image originale ;
- pour chaque objet d’une classe affinée : **crop** de la bbox (avec padding optionnel, défaut 5 %) ;
- classification du crop avec la logique V1 (`INCONNU` / `INCERTAIN`) ;
- les classes non affinées restent des détections simples (`Carrot 93 %`).

Stockage : `pipelines/*.json` sous le data root (`VISIONTRAINER_DATA_DIR` ou `./artifacts`). Les fichiers référencent des **run id** (pas les poids embarqués), ce qui reste portable après redémarrage Docker.

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
- `pipelines/` — configs Détection → Classification
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
