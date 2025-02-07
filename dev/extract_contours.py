"""
医学图像有效区域提取

功能：
1. 从指定目录读取原始图像
2. 提取图像凸包并裁剪有效区域
3. 保存裁剪后的灰度图像和轮廓数据

使用示例：
python dev/extract_contours.py \
    --input data/raw/Training_Dataset \
    --output data/processed/cropped_images
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


def find_convex_hull(image: np.ndarray) -> np.ndarray:
    """寻找图像凸包

    Args:
        image: 输入图像，BGR格式

    Returns:
        凸包坐标点集合
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("未检测到有效轮廓")
    max_contour = max(contours, key=lambda x: x.shape[0])
    return cv2.convexHull(max_contour)


def crop_to_convex_hull(image: np.ndarray, hull: np.ndarray) -> tuple:
    """根据凸包裁剪图像

    Args:
        image: 输入图像，BGR格式
        hull: 凸包坐标点集合

    Returns:
        tuple: (裁剪后的图像, 裁剪区域的偏移量(x, y))
    """
    x, y, w, h = cv2.boundingRect(hull)
    return image[y : y + h, x : x + w], (x, y)


def process_image(image: np.ndarray) -> Dict[str, Any]:
    """图像处理流水线

    Args:
        image: 输入图像，BGR格式

    Returns:
        包含裁剪图像和轮廓数据的字典
    """
    hull = find_convex_hull(image)
    cropped_image, (offset_x, offset_y) = crop_to_convex_hull(image, hull)
    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # 转换轮廓坐标为裁剪后图像的坐标系
    contour_cropped = (hull - np.array([offset_x, offset_y])).squeeze().tolist()

    return {
        "width": gray_image.shape[1],
        "height": gray_image.shape[0],
        "contour": contour_cropped,
        "gray_image": gray_image,
    }


def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="医学图像有效区域提取")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/Training_Dataset"),
        help="原始图像目录路径（默认：data/raw/Training_Dataset）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/cropped_images"),
        help="输出目录路径（默认：data/processed/cropped_images）",
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

                # 处理图像
                processed_data = process_image(image)

                # 保存灰度图像
                output_path = args.output / img_path.name
                output_path.parent.mkdir(parents=True, exist_ok=True)
                # 确保保存为单通道灰度JPEG
                cv2.imwrite(
                    str(output_path),
                    processed_data["gray_image"],
                    [
                        int(cv2.IMWRITE_JPEG_QUALITY),
                        90,
                        int(cv2.IMWRITE_JPEG_OPTIMIZE),
                        1,
                    ],
                )

                # 创建ContourImage实例并添加到数据集
                contour_image = ContourImage(
                    filename=img_path.name,
                    width=processed_data["width"],
                    height=processed_data["height"],
                    contour=processed_data["contour"],
                )
                dataset.root.append(contour_image)

            except (IOError, ValueError, ValidationError) as e:
                progress.console.print(f"[red]错误[/red] {img_path.name}: {str(e)}")
            finally:
                progress.update(task, advance=1)

    # 保存所有轮廓数据到单个文件
    contour_path = args.output.parent / "cropped_contours.json"
    with contour_path.open("w") as f:
        f.write(dataset.model_dump_json(indent=None))

    console.print(
        f"\n处理完成：共处理 {len(image_files)} 张图像，"
        f"成功 {len(dataset)} 张，保存至 {args.output}"
    )


if __name__ == "__main__":
    main()
