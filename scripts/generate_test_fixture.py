"""Generate the committed valid YOLO test fixture."""

from __future__ import annotations

from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "valid_dataset"


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)

    data_yaml = {
        "path": str(ROOT),
        "train": "train/images",
        "val": "val/images",
        "names": {0: "cat", 1: "dog"},
    }
    (ROOT / "data.yaml").write_text(yaml.dump(data_yaml), encoding="utf-8")

    samples = [
        ("train/images/cat_1.jpg", (220, 80, 80), "train/labels/cat_1.txt", "0 0.5 0.5 0.5 0.5"),
        ("train/images/dog_1.jpg", (80, 180, 80), "train/labels/dog_1.txt", "1 0.35 0.35 0.3 0.3"),
        ("val/images/cat_2.jpg", (80, 80, 220), "val/labels/cat_2.txt", "0 0.55 0.55 0.35 0.35"),
    ]

    for image_rel, color, label_rel, label_line in samples:
        image_path = ROOT / image_rel
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (128, 128), color).save(image_path)

        label_path = ROOT / label_rel
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(label_line + "\n", encoding="utf-8")

    print(f"Fixture created at {ROOT}")


if __name__ == "__main__":
    main()
