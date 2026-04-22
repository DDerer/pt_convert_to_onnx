#!/usr/bin/env python3
"""Run detection on 1.png using model.pt."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect with model.pt")
    parser.add_argument("--model", type=Path, default=Path("models/model.pt"), help="Path to .pt model")
    parser.add_argument("--image", type=Path, default=Path("1.png"), help="Path to image")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_root = project_root / "runs_compare"
    model_path = args.model if args.model.is_absolute() else project_root / args.model
    image_path = args.image if args.image.is_absolute() else project_root / args.image

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ultralytics is not installed. Install with: pip install ultralytics"
        ) from exc

    model = YOLO(str(model_path))
    results = model.predict(
        source=str(image_path),
        imgsz=args.imgsz,
        conf=args.conf,
        save=True,
        project=str(output_root),
        name="pt",
        exist_ok=True,
        verbose=False,
    )

    r = results[0]
    print(f"[PT] Image: {image_path}")
    print(f"[PT] Detections: {len(r.boxes)}")

    names = model.names
    for i, box in enumerate(r.boxes, start=1):
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        xyxy = box.xyxy[0].tolist()
        cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
        print(
            f"[PT] {i}: class={cls_name} conf={conf:.4f} "
            f"xyxy=({xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f})"
        )

    print(f"[PT] Saved image: {output_root / 'pt' / (image_path.stem + '.jpg')}")


if __name__ == "__main__":
    main()
