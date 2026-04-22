from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_cfg = project_root / "robocup.yaml"
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(data_cfg),
        epochs=100,
        imgsz=640,
        batch=8,
        workers=2,
        device=0,
        project=str(project_root / "runs"),
        name="robocup_yolov8n",
    )


if __name__ == "__main__":
    main()
