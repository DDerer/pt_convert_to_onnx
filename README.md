# tranfor_onnx

这是一个示例项目：把 YOLO 的 `.pt` 模型转换为 `.onnx`，并用同一张图片分别用 PT/ONNX 做检测对比。

## 目录结构

```text
.
├── 1.png
├── convert_to_onnx.py
├── detect
│   ├── detect_with_onnx.py
│   └── detect_with_pt.py
├── models
│   ├── model.onnx
│   └── model.pt
├── README.md
├── requirements.txt
├── runs_compare
│   ├── onnx
│   │   └── 1.jpg
│   └── pt
│       └── 1.jpg
└── train.yaml
```

## 环境准备

建议 Python 3.10+。

```bash
python -m venv yolo_env
source yolo_env/bin/activate
pip install -r requirements.txt
```

## 1) PT 转 ONNX

默认输入权重是 `models/model.pt`，默认输出是同目录同名的 `models/model.onnx`。

```bash
python convert_to_onnx.py
```

显式指定路径示例：

```bash
python convert_to_onnx.py --weights models/model.pt --output models/model.onnx --imgsz 640 --opset 12 --device cpu
```

常用参数：

- `--weights`：输入 `.pt` 权重路径
- `--output`：输出 `.onnx` 路径
- `--imgsz`：导出尺寸
- `--batch`：导出 batch
- `--opset`：ONNX opset 版本
- `--device`：`cpu` 或 GPU 编号（如 `0`）
- `--dynamic`：启用动态维度
- `--simplify`：导出后简化 ONNX

## 2) PT 模型检测

```bash
python detect/detect_with_pt.py --model models/model.pt --image 1.png --imgsz 640 --conf 0.25
```

结果图输出到：`runs_compare/pt/1.jpg`

## 3) ONNX 模型检测

```bash
python detect/detect_with_onnx.py --model models/model.onnx --image 1.png --imgsz 640 --conf 0.25 --device cpu
```

结果图输出到：`runs_compare/onnx/1.jpg`

## 4) 对比结果

直接对比下面两张图：

- `runs_compare/pt/1.jpg`
- `runs_compare/onnx/1.jpg`

如果框位置、类别和置信度接近，说明 ONNX 导出和推理基本正常。

## 常见问题

1. `ModuleNotFoundError: ultralytics`

```bash
pip install ultralytics
```

2. `ModuleNotFoundError: onnxruntime`

```bash
pip install onnxruntime
```

3. 导出时提示缺少 `onnx`

```bash
pip install "onnx>=1.12,<2.0.0"
```

4. CPU 导出时不要加 `--half`

- `--half` 通常用于 GPU FP16，CPU 下常会报错或没有收益。

## 说明

- 当前脚本已按“项目根目录”解析相对路径，所以你在项目根目录或其他目录执行都可以。
