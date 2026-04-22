#!/usr/bin/env python3
"""Export a YOLO .pt model to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a YOLO .pt model to ONNX."
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=Path,
        default=Path("best.pt"),
        help="Path to the .pt model (default: best.pt).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output ONNX filename (default: same stem as weights).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for export (default: 640).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size used by exporter (default: 1).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version (default: 12).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Export device, e.g. cpu or 0 (default: cpu).",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export FP16 model.",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic axes in exported ONNX.",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Run ONNX simplifier after export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ultralytics is not installed. Install it with: pip install ultralytics"
        ) from exc

    if not args.weights.exists():
        raise FileNotFoundError(f"Weights file not found: {args.weights}")

    if args.imgsz <= 0:
        raise ValueError("--imgsz must be a positive integer")

    if args.batch <= 0:
        raise ValueError("--batch must be a positive integer")

    if args.half and str(args.device).lower() == "cpu":
        raise ValueError("--half is usually not supported on CPU export; use GPU device.")

    output_name = args.output or f"{args.weights.stem}.onnx"

    print(f"Loading model: {args.weights}")
    model = YOLO(str(args.weights))

    print("Exporting to ONNX...")
    exported_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        batch=args.batch,
        opset=args.opset,
        device=args.device,
        half=args.half,
        dynamic=args.dynamic,
        simplify=args.simplify,
        name=output_name,
    )

    print(f"Export finished: {exported_path}")


if __name__ == "__main__":
    main()
