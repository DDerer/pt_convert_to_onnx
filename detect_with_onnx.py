#!/usr/bin/env python3
"""Run detection on 1.png using best.onnx."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect with best.onnx")
    parser.add_argument("--model", type=Path, default=Path("best.onnx"), help="Path to .onnx model")
    parser.add_argument("--image", type=Path, default=Path("1.png"), help="Path to image")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cpu", help="Inference device, default cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path.cwd() / "runs_compare"

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ultralytics is not installed. Install with: pip install ultralytics"
        ) from exc
    try:
        import onnxruntime  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "onnxruntime is not installed. Install with: pip install onnxruntime"
        ) from exc

    model = YOLO(str(args.model), task="detect")
    results = model.predict(
        source=str(args.image),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=True,
        project=str(output_root),
        name="onnx",
        exist_ok=True,
        verbose=False,
    )

    r = results[0]
    print(f"[ONNX] Image: {args.image}")
    print(f"[ONNX] Detections: {len(r.boxes)}")

    names = model.names
    for i, box in enumerate(r.boxes, start=1):
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        xyxy = box.xyxy[0].tolist()
        cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
        print(
            f"[ONNX] {i}: class={cls_name} conf={conf:.4f} "
            f"xyxy=({xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f})"
        )

    print("[ONNX] Saved image: runs_compare/onnx/1.png")


if __name__ == "__main__":
    main()
