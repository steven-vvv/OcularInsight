from pathlib import Path

import cv2
import numpy as np
from rich import print

try:
    from src.data.schemas.contour import ContourDataset
except ModuleNotFoundError:
    import pathlib
    import sys

    sys.path.append(str(pathlib.Path("./")))
    from src.data.schemas.contour import ContourDataset


def main():
    """主函数：浏览轮廓处理结果"""
    # 读取轮廓 JSON 文件
    json_path = Path(r"./data/processed/cropped_contours.json")
    if not json_path.exists():
        print(f"[red]错误：未找到轮廓文件 {json_path}")
        return

    # 加载轮廓数据集
    with open(json_path, "r", encoding="utf-8") as f:
        dataset = ContourDataset.model_validate_json(f.read())

    print(f"[green]成功加载 {len(dataset)} 张图像的轮廓数据")

    # 遍历并显示每张图像
    for i, image_data in enumerate(dataset):
        print(f"[blue]正在显示第 {i + 1}/{len(dataset)} 张图像: {image_data.filename}")

        # 读取图像
        img_path = Path(r"./data/processed/cropped_images") / image_data.filename
        if not img_path.exists():
            print(f"[yellow]警告：未找到图像文件 {img_path}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[yellow]警告：无法读取图像文件 {img_path}")
            continue

        # 计算缩放比例
        scale = 1000 / max(image_data.width, image_data.height)
        new_width = int(image_data.width * scale)
        new_height = int(image_data.height * scale)
        img = cv2.resize(img, (new_width, new_height))

        # 绘制轮廓
        contour = (np.array(image_data.contour) * scale).astype(np.int32)
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)

        # 显示图像
        cv2.imshow("Contour Viewer", img)
        key = cv2.waitKey(0)
        if key in (ord("q"), ord("Q"), 27):  # q/Q/Esc 退出
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
