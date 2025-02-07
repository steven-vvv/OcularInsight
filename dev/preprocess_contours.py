# ocular-insight/scripts/preprocess_contours.py
"""
医学图像轮廓预处理流水线

功能：
1. 从指定目录读取原始图像
2. 自动去除黑边并提取有效区域轮廓
3. 验证并保存结构化轮廓数据

使用示例：
python scripts/preprocess_contours.py \
    --input data/raw/Training_Dataset \
    --output data/processed/contours.json
"""

import argparse
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
from pydantic import ValidationError
from rich.console import Console
from rich.progress import Progress

try:
    from src.data.schemas.contour import ContourDataset, ContourImage
except ModuleNotFoundError:
    import pathlib
    import sys

    sys.path.append(str(pathlib.Path("./")))
    from src.data.schemas.contour import ContourDataset, ContourImage


def resize_long_edge(image: np.ndarray, long_edge: int) -> tuple:
    """等比例缩放图像（以长边为基准）

    Args:
        image: 输入图像，BGR格式
        long_edge: 目标长边长度

    Returns:
        tuple: (缩放后图像, 缩放比例)
    """
    h, w = image.shape[:2]
    scale = long_edge / max(h, w)
    resized_image = cv2.resize(image, (int(w * scale), int(h * scale)))
    return resized_image, scale


def find_largest_contour(image: np.ndarray) -> np.ndarray:
    """寻找最大轮廓（基于预处理后的二值图像）

    Args:
        image: 输入图像，BGR格式

    Returns:
        最大轮廓的坐标点集合

    Raises:
        ValueError: 未找到任何轮廓时抛出
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    padded = cv2.copyMakeBorder(gray, 5, 5, 0, 0, cv2.BORDER_CONSTANT, value=[0])
    _, binary = cv2.threshold(padded, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("未检测到有效轮廓")
    return max(contours, key=lambda x: x.shape[0])


def process_image(image: np.ndarray) -> Dict[str, Any]:
    """完整图像处理流水线

    Args:
        image: 输入图像，BGR格式

    Returns:
        包含原始尺寸和轮廓数据的字典
    """
    h, w = image.shape[:2]
    resized_img, scale = resize_long_edge(image, 250)
    max_contour = find_largest_contour(resized_img)
    hull = cv2.convexHull(max_contour)
    contour_original = hull.astype(np.float64)
    contour_original += np.array([1, -4])
    contour_original /= scale

    # 裁剪到图像边界
    contour_original[..., 0] = np.clip(contour_original[..., 0], 0, w - 1)
    contour_original[..., 1] = np.clip(contour_original[..., 1], 0, h - 1)

    return {
        "width": w,
        "height": h,
        "contour": contour_original.astype(int).squeeze().tolist(),
    }


def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="医学图像轮廓预处理")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/Training_Dataset"),
        help="原始图像目录路径（默认：data/raw/Training_Dataset）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/contours.json"),
        help="轮廓数据输出路径（默认：data/processed/contours.json）",
    )
    args = parser.parse_args()

    console = Console()
    valid_extensions = (".jpg", ".jpeg", ".png")
    dataset = ContourDataset(root=[])

    # 扫描输入目录
    image_files = [
        f for f in args.input.iterdir() if f.suffix.lower() in valid_extensions
    ]
    if not image_files:
        console.print(f"[red]错误：目录 {args.input} 中未找到图像文件[/red]")
        return

    # 处理进度跟踪
    with Progress() as progress:
        task = progress.add_task("正在处理图像...", total=len(image_files))

        for img_path in image_files:
            try:
                # 读取图像
                image = cv2.imread(str(img_path))
                if image is None:
                    raise IOError("无效的图像文件")

                # 处理图像并验证数据
                raw_data = process_image(image)
                contour_image = ContourImage(filename=img_path.name, **raw_data)
                dataset.root.append(contour_image)

            except (IOError, ValueError, ValidationError) as e:
                progress.console.print(f"[red]错误[/red] {img_path.name}: {str(e)}")
            finally:
                progress.update(task, advance=1)

    # 保存结果
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        f.write(dataset.model_dump_json(indent=None))

    console.print(
        f"\n处理完成：共处理 {len(image_files)} 张图像，"
        f"成功 {len(dataset)} 张，保存至 {args.output}"
    )


if __name__ == "__main__":
    main()
