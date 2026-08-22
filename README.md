# Vision Trainer

Application locale pour préparer et entraîner des modèles de détection d'objets (YOLO / Ultralytics).

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Lancement

```bash
streamlit run app/streamlit_app.py
```

## Tests

```bash
pytest
```

## Artefacts d'entraînement

Les runs sont écrits dans `artifacts/runs/<run_id>/` (ignoré par git).
