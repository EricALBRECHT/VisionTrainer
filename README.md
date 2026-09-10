# Vision Trainer

Application locale pour entraîner des modèles **YOLO / Ultralytics** :

- **Détection** — objets + bounding boxes (*où ?*)
- **Classification** — classe / probabilités (*quoi ?*)
- **Segmentation** — masques / polygones (*quels pixels ?*)
- **Pipelines** — Détection → Classification optionnelle

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

## Segmentation

**Classification** : quoi ? · **Détection** : où ? · **Segmentation** : quels pixels ?

### Dataset

Format YOLO segmentation Ultralytics :

```text
dataset/
  train/images/
  train/labels/   # class x1 y1 x2 y2 x3 y3 ...
  val/images/
  val/labels/
  data.yaml
```

Les labels sont des **polygones** normalisés. VisionTrainer **refuse** les fichiers au format détection (`class xc yc w h`) sans conversion automatique.

Import ZIP sur **Segmentation — Dataset**, avec prévisualisation des polygones.

### Entraînement

Modèles proposés (Ultralytics 8.4.x) : `yolo11n-seg.pt` (défaut), `yolo11s-seg.pt`, `yolo11m-seg.pt`.

Batch **auto** recommandé (VRAM plus élevée qu’en détection).

### Inférence

Page **Inférence** → mode **Segmentation** : masques semi-transparents, contours, labels, surface en **pixels** (+ ratio d’image). Export JSON des polygones possible.

### Métriques

L’historique distingue **Box** (Precision / Recall / mAP) et **Mask** (mêmes indicateurs sur les masques). Ce ne sont pas les mêmes qualités.

## Inférence classification — seuil « Inconnu »

Sur la page Inférence (mode Classification) :

- **seuil de confiance minimum** → sous le seuil : `INCONNU / CONFIANCE INSUFFISANTE`
- **écart Top-1 / Top-2** → trop faible : `INCERTAIN`

Ce n’est **pas** une détection OOD formelle.

## Pipelines

Chaînage optionnel **classe par classe** :

```text
Détection
    ↓
Classification optionnelle
    ↓
Segmentation optionnelle
```

Chaque classe du détecteur peut activer indépendamment :

- un **classificateur** (`task=classify`) — Top-N, INCONNU / INCERTAIN ;
- un **segmenter** (`task=segment`) — masques sur le crop, polygones remappés en coordonnées image.

Exemple : `Bearing` → classifier type → segmenter défaut ; `Other` → détection seule.

Stockage : `pipelines/*.json` (format **v2** à la sauvegarde). Les pipelines **v1** (detect→classify) restent lisibles sans réécriture automatique.

Surfaces de masque : **% du crop analysé** (et optionnellement % de l'image) — pas une mesure physique.

## Vidéo & Caméra

Traitement **frame par frame** réutilisant les moteurs image :

- modes : **Détection**, **Segmentation**, **Pipeline** ;
- export vidéo annotée (MP4, **sans audio** en V1) ;
- **FPS source / export** ≠ **FPS traitement** (vitesse d'inférence) ;
- `frame stride` : frames non analysées = image originale (pas d'ancienne annotation) ;
- modèles chargés **une fois** (cache) avant la boucle ;
- caméra : indices `0, 1, …` côté **serveur** (pas la webcam navigateur).

### Limitations webcam Docker / WSL

`cv2.VideoCapture(index)` voit les périphériques Linux du conteneur / WSL, pas automatiquement la webcam Windows. En cas d'échec : utiliser un fichier vidéo, ou monter `/dev/video*` / lancer hors Docker.

### Préparation future

Même API frame (`processor.process(frame)`) pour RTSP, robot, API — résultat structuré séparé du rendu annoté.

## Lancement avec Docker

Prérequis : Docker et Docker Compose.

```bash
docker compose up --build
```

Puis : http://localhost:8501

Upload Streamlit : `maxUploadSize = 10240` (Mo, soit 10 Go) dans `.streamlit/config.toml`.

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
