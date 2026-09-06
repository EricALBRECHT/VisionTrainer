# Vision Trainer

Application locale pour préparer et entraîner des modèles de détection d'objets (YOLO / Ultralytics).

## Installation (local)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Lancement (local)

```bash
streamlit run app/streamlit_app.py
```

Données locales par défaut : `./artifacts/` (`datasets/`, `runs/`).

## Lancement avec Docker

Prérequis : Docker et Docker Compose.

```bash
git clone <repository>
cd vision-trainer
docker compose up --build
```

Puis ouvrir :

```text
http://localhost:8501
```

### Arrêter

```bash
docker compose down
```

### Reconstruire après modification

```bash
docker compose up --build
```

### Voir les logs

```bash
docker compose logs -f
```

### Données persistantes

Le bind mount `./data` (hôte) → `/app/data` (conteneur) conserve :

- `data/datasets/` — ZIP importés et datasets extraits
- `data/runs/` — runs d'entraînement, checkpoints, `best.*`
- `data/ultralytics/` — cache / config Ultralytics
- `data/tmp/` — fichiers temporaires d'inférence

Ces dossiers survivent à `docker compose down` et à une reconstruction d'image.

Variable d'environnement optionnelle :

```text
VISIONTRAINER_DATA_DIR=/app/data
```

Sans cette variable (hors Docker), l'application utilise `./artifacts`.


### Docker Desktop + WSL

Si le montage `./data` échoue (erreur *distro mount service*), lancez :

```bash
docker compose -f docker-compose.yml -f docker-compose.wsl.yml up --build
```

Les données sont alors dans le volume Docker `vision-trainer-data`.


## Docker CPU

```bash
docker compose up --build
```

Puis : http://localhost:8501

Sur Docker Desktop + WSL si le bind-mount `./data` échoue :

```bash
docker compose -f docker-compose.yml -f docker-compose.wsl.yml up --build
```

## Docker GPU NVIDIA

Prérequis :

- GPU NVIDIA + drivers à jour
- Linux : [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Windows : Docker Desktop avec support GPU WSL2 activé

Lancement :

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Avec le fallback volume WSL si besoin :

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.wsl.yml up --build
```

Vérifier CUDA dans le conteneur :

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml exec vision-trainer \
  python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
```

La configuration CPU (`docker compose up --build`) reste utilisable sans GPU.

## Tests

```bash
pytest
```

## Artefacts d'entraînement

Les runs sont écrits sous `<data_root>/runs/<run_id>/` (ignoré par git).
