"""Excel标注数据转换脚本

本脚本用于将原始Excel格式的眼底图像标注数据转换为标准化的CSV格式，
同时进行数据验证和错误记录，确保数据质量符合项目要求。

主要功能：
1. 读取指定Excel文件中的标注数据
2. 验证图像文件存在性
3. 使用Pydantic模型进行数据验证
4. 生成标准化的CSV文件
5. 记录转换过程中的错误信息

使用说明：
1. 确保当前工作目录为项目根目录
2. 运行脚本：python scripts/convert_excel_to_csv.py
   或指定路径：python scripts/convert_excel_to_csv.py --excel_path data/raw/Training_Dataset.xlsx --image_dir data/raw/Training_Dataset
"""

import argparse
from pathlib import Path
from typing import List, Tuple
import sys

import pandas as pd
from rich import print

# 项目根目录路径处理
PROJECT_ROOT = Path.cwd()
sys.path.append(str(PROJECT_ROOT))

from src.data.schemas.eye_diagnosis import EyeDiagnosisLabel  # noqa: E402


def validate_image_paths(raw_dir: Path, left_path: str, right_path: str) -> Tuple[Path, Path]:
    """验证眼底图像文件路径是否存在

    Args:
        raw_dir: 原始数据目录路径
        left_path: 左眼图像相对路径
        right_path: 右眼图像相对路径

    Returns:
        Tuple[Path, Path]: 验证通过的左右眼图像绝对路径

    Raises:
        FileNotFoundError: 当图像文件不存在时
    """
    left_abs = raw_dir / left_path
    right_abs = raw_dir / right_path

    if not left_abs.exists():
        raise FileNotFoundError(f"左眼图像文件不存在: {left_abs}")
    if not right_abs.exists():
        raise FileNotFoundError(f"右眼图像文件不存在: {right_abs}")

    return left_abs, right_abs


def convert_excel_to_csv(
    excel_path: Path,
    image_dir: Path,
    output_csv: Path
) -> None:
    """Excel标注数据转换主流程

    Args:
        excel_path: 输入Excel文件路径
        image_dir: 图像文件存储目录
        output_csv: 输出CSV文件路径

    Raises:
        FileNotFoundError: 当输入文件不存在时
        ValueError: 当数据验证失败时
    """
    # 确保输出目录存在
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # 读取原始数据
    try:
        df = pd.read_excel(excel_path, sheet_name="Sheet1", dtype={"N": int})
    except Exception as e:
        print(f"[bold red]Excel读取失败: {e}[/bold red]")
        raise

    valid_records: List[dict] = []
    errors: List[Tuple[int, str]] = []

    # 数据转换流水线
    for idx, row in df.iterrows():
        try:
            # 基础字段转换
            record = {
                "id": int(row["ID"]),
                "patient_age": int(row["Patient Age"]),
                "patient_sex": row["Patient Sex"],
                "left_fundus": row["Left-Fundus"],
                "right_fundus": row["Right-Fundus"],
                "left_diagnostic_keywords": row["Left-Diagnostic Keywords"],
                "right_diagnostic_keywords": row["Right-Diagnostic Keywords"],
                "N": bool(row["N"]),
                "D": bool(row["D"]),
                "G": bool(row["G"]),
                "C": bool(row["C"]),
                "A": bool(row["A"]),
                "H": bool(row["H"]),
                "M": bool(row["M"]),
                "O": bool(row["O"]),
            }

            # 图像路径验证
            validate_image_paths(
                image_dir,
                record["left_fundus"],
                record["right_fundus"]
            )

            # Pydantic模型验证
            label = EyeDiagnosisLabel(**record)
            valid_records.append(label.model_dump(by_alias=True))

        except Exception as e:
            errors.append((idx + 2, str(e)))  # +2考虑Excel行号偏移

    # 结果持久化
    if valid_records:
        pd.DataFrame(valid_records).to_csv(output_csv, index=False)
        print(f"[bold green]成功转换 {len(valid_records)} 条记录至 {output_csv}[/bold green]")
    else:
        print("[bold red]无有效数据需要转换[/bold red]")

    # 错误日志记录
    if errors:
        error_log = output_csv.parent / "conversion_errors.log"
        with open(error_log, "w") as f:
            f.write("行号\t错误信息\n")
            for line, err in errors:
                f.write(f"{line}\t{err}\n")
        print(f"[bold yellow]发现 {len(errors)} 处错误，详见 {error_log}[/bold yellow]")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Excel标注数据转换工具")
    parser.add_argument(
        "--excel_path",
        type=Path,
        help="输入Excel文件路径"
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        help="图像文件存储目录"
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        help="输出CSV文件路径"
    )
    return parser.parse_args()


if __name__ == "__main__":
    # 硬编码默认路径
    DEFAULT_PATHS = {
        "excel_path": PROJECT_ROOT / "data" / "raw" / "Training_Dataset.xlsx",
        "image_dir": PROJECT_ROOT / "data" / "raw" / "Training_Dataset",
        "output_csv": PROJECT_ROOT / "data" / "processed" / "training_labels.csv",
    }

    # 解析命令行参数
    args = parse_args()

    # 使用提供的路径或默认路径
    a_excel_path = args.excel_path or DEFAULT_PATHS["excel_path"]
    a_image_dir = args.image_dir or DEFAULT_PATHS["image_dir"]
    a_output_csv = args.output_csv or DEFAULT_PATHS["output_csv"]

    # 执行转换
    convert_excel_to_csv(a_excel_path, a_image_dir, a_output_csv)
