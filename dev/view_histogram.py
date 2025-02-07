"""使用OpenCV和Matplotlib分析图像直方图。

此脚本用于加载图像和轮廓数据，并显示图像有效区域的像素分布直方图。
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from rich import print

try:
    from src.data.schemas.contour import ContourDataset
except ModuleNotFoundError:
    import pathlib
    import sys

    sys.path.append(str(pathlib.Path("./")))
    from src.data.schemas.contour import ContourDataset


@dataclass
class ImageAnalysis:
    """图像分析结果数据类。"""

    title: str
    image: np.ndarray  # 图像数据
    pixels: np.ndarray  # 有效区域的像素值
    color: str = "b"  # 绘图颜色


def load_contours(json_path: Path) -> ContourDataset:
    """加载轮廓数据。

    Args:
        json_path: 轮廓JSON文件路径。

    Returns:
        轮廓数据集对象。

    Raises:
        FileNotFoundError: 当轮廓文件不存在时。
        ValueError: 当轮廓文件无法解析时。
    """
    if not json_path.exists():
        raise FileNotFoundError(f"未找到轮廓文件：{json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        dataset = ContourDataset.model_validate_json(f.read())

    print(f"[green]成功加载 {len(dataset)} 张图像的轮廓数据")
    return dataset


def create_mask(image_shape: tuple[int, int], contour: list[list[int]]) -> np.ndarray:
    """创建图像掩码。

    Args:
        image_shape: 图像形状 (高, 宽)。
        contour: 轮廓点列表。

    Returns:
        与图像同形状的二值掩码。
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    contour_array = np.array(contour, dtype=np.int32)
    cv2.drawContours(mask, [contour_array], -1, [255], -1)
    return mask


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """使用全局直方图均衡化处理图像。

    Args:
        image: 输入图像数组（单通道）

    Returns:
        均衡化后的图像数组（uint8类型）

    Raises:
        ValueError: 输入不是单通道图像时抛出
    """
    if len(image.shape) != 2:
        raise ValueError("仅支持单通道图像输入")

    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    return cv2.equalizeHist(image)


def unsharp_masking(
    image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0, amount: float = 1.0
) -> np.ndarray:
    """使用非锐化掩模增强图像细节。

    Args:
        image: 输入图像数组（单通道）
        kernel_size: 高斯核大小，必须是奇数
        sigma: 高斯核标准差
        amount: 锐化强度

    Returns:
        增强后的图像数组（uint8类型）

    Raises:
        ValueError: kernel_size不是奇数时抛出
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size必须是奇数")

    # 创建高斯模糊版本
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    # 计算非锐化掩模
    mask = image.astype(np.float32) - blurred.astype(np.float32)

    # 增强图像
    sharpened = image.astype(np.float32) + amount * mask

    # 裁剪到有效范围
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def plot_image_analysis(
    analyses: List[ImageAnalysis], suptitle: str = "Pixel Distribution Analysis"
):
    """绘制多组图像分析结果。

    Args:
        analyses: 图像分析结果列表
        suptitle: 总标题
    """
    n_cols = len(analyses)  # 每个分析结果占一列
    n_rows = 3  # 每列显示：图像、PDF、CDF
    figsize = (5 * n_cols, 12)  # 根据列数调整图像大小

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.3, wspace=0.3)

    # 计算所有分析的bin参数
    bins = 256
    bin_centers = np.arange(bins)
    hist_range = (0, 256)

    for col, analysis in enumerate(analyses):
        # 准备显示图像
        display_img = analysis.image.copy()

        # 1. 显示图像
        ax_img = fig.add_subplot(gs[0, col])
        ax_img.imshow(display_img, cmap="gray")
        ax_img.set_title(analysis.title)
        ax_img.axis("off")

        # 2. 显示PDF（直方图）
        ax_pdf = fig.add_subplot(gs[1, col])
        hist, _ = np.histogram(
            analysis.pixels, bins=bins, range=hist_range, density=True
        )
        ax_pdf.bar(bin_centers, hist, width=1.0, color=analysis.color, alpha=0.7)
        ax_pdf.set_title("PDF")
        ax_pdf.set_xlabel("Pixel Value")
        ax_pdf.set_ylabel("Density")
        ax_pdf.grid(True, alpha=0.3)

        # 3. 显示CDF
        ax_cdf = fig.add_subplot(gs[2, col])
        cdf = np.cumsum(hist) / np.sum(hist)
        ax_cdf.plot(bin_centers, cdf, f"{analysis.color}-", lw=2)
        ax_cdf.set_title("CDF")
        ax_cdf.set_xlabel("Pixel Value")
        ax_cdf.set_ylabel("Cumulative Probability")
        ax_cdf.grid(True, alpha=0.3)

    plt.suptitle(suptitle, y=0.95)
    plt.show()


def main():
    """主函数：分析图像直方图。"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="分析图像直方图")
    parser.add_argument(
        "--image",
        type=str,
        default=r"./data/processed/cropped_images/2217_left.jpg",
        help="图像文件路径",
    )
    parser.add_argument(
        "--contours",
        type=str,
        default=r"./data/processed/cropped_contours.json",
        help="轮廓JSON文件路径",
    )
    args = parser.parse_args()

    # 加载轮廓数据
    contours = load_contours(Path(args.contours))

    # 读取图像
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"[red]错误：未找到图像文件 {img_path}")
        return

    # 读取为灰度图像
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[red]错误：无法读取图像文件 {img_path}")
        return

    # 查找对应的轮廓数据
    image_data = next(
        (data for data in contours if data.filename == img_path.name), None
    )
    if image_data is None:
        print(f"[red]错误：未找到图像 {img_path.name} 的轮廓数据")
        return

    # 验证图像尺寸
    if img.shape != (image_data.height, image_data.width):
        print("[red]错误：图像尺寸不匹配[/]")
        print(f"预期尺寸: {image_data.width}x{image_data.height}")
        print(f"实际尺寸: {img.shape[1]}x{img.shape[0]}")
        return

    # 创建掩码
    mask = create_mask(img.shape, image_data.contour)

    # 获取有效区域
    valid_area = cv2.bitwise_and(img, mask)
    valid_pixels = valid_area[mask > 0]
    valid_area_display = valid_area.copy()
    valid_area_display[mask == 0] = 255

    # 直方图均衡化
    equalized_area = histogram_equalization(valid_area)
    equalized_pixels = equalized_area[mask > 0]
    equalized_display = equalized_area.copy()
    equalized_display[mask == 0] = 255

    # 对均衡化后的图像应用非锐化掩模
    sharpened_area = unsharp_masking(
        equalized_area, kernel_size=9, sigma=1.0, amount=1.0
    )
    sharpened_pixels = sharpened_area[mask > 0]
    sharpened_display = sharpened_area.copy()
    sharpened_display[mask == 0] = 255

    # 创建分析结果
    analyses = [
        ImageAnalysis("Original", valid_area_display, valid_pixels, "b"),
        ImageAnalysis("Equalized", equalized_display, equalized_pixels, "r"),
        ImageAnalysis("Sharpened", sharpened_display, sharpened_pixels, "g"),
    ]

    # 显示分析结果
    plot_image_analysis(analyses, img_path.name)


if __name__ == "__main__":
    main()
