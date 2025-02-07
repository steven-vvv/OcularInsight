# ocular-insight/src/data/schemas/contour.py
from typing import Annotated, List

from pydantic import BaseModel, Field, RootModel, model_validator
from pydantic.functional_validators import AfterValidator


def validate_point(value: List[int]) -> List[int]:
    """验证单个轮廓点的坐标格式

    Args:
        value: 待验证的坐标点列表，应为两个元素的列表

    Raises:
        ValueError: 当坐标点不符合要求时抛出

    Returns:
        验证通过的坐标点列表
    """
    if len(value) != 2:
        raise ValueError("坐标点必须包含两个元素")
    if value[0] < 0 or value[1] < 0:
        raise ValueError("坐标值不能为负数")
    return value


ContourPoint = Annotated[
    List[int],
    Field(
        min_length=2,
        max_length=2,
        description="单个轮廓点的坐标 [x, y]，坐标值为非负整数",
    ),
    AfterValidator(validate_point),
]

ImageDimension = Annotated[
    int, Field(gt=0, strict=True, description="图像尺寸（宽度/高度），必须为正整数")
]


class ContourImage(BaseModel):
    """单张图像的轮廓数据模型

    Attributes:
        filename: 图像文件名，支持.jpg/.jpeg/.png格式
        width: 图像原始宽度（像素）
        height: 图像原始高度（像素）
        contour: 有效区域的轮廓点坐标列表
    """

    filename: Annotated[
        str,
        Field(
            pattern=r"\.(jpg|jpeg|png)$",
            description="图像文件名，需以.jpg/.jpeg/.png结尾",
        ),
    ]
    width: ImageDimension
    height: ImageDimension
    contour: List[ContourPoint]

    @model_validator(mode="after")
    def check_contour_bounds(self) -> "ContourImage":
        """验证轮廓坐标是否在图像边界内

        Raises:
            ValueError: 当轮廓点超出图像范围时抛出
        """
        max_x = self.width - 1
        max_y = self.height - 1

        for point in self.contour:
            if point[0] > max_x:
                raise ValueError(f"X坐标 {point[0]} 超出图像宽度 {max_x}")
            if point[1] > max_y:
                raise ValueError(f"Y坐标 {point[1]} 超出图像高度 {max_y}")
        return self


class ContourDataset(RootModel):
    """轮廓数据集模型

    直接使用ContourImage列表作为根模型，支持迭代和索引访问

    Example:
        >>> dataset = ContourDataset.model_validate_json(...)
        >>> for image_data in dataset:
        >>>     print(image_data.filename)
    """

    root: List[ContourImage]

    def __iter__(self):
        """支持迭代"""
        return iter(self.root)

    def __len__(self):
        """支持len()函数"""
        return len(self.root)

    def __getitem__(self, index):
        """支持索引访问"""
        return self.root[index]

    model_config = {
        "json_schema_extra": {
            "example": [
                {
                    "filename": "sample.jpg",
                    "width": 1024,
                    "height": 768,
                    "contour": [[100, 200], [300, 400]],
                }
            ]
        }
    }
