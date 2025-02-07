from typing import Annotated, Literal
from pydantic import BaseModel, Field, AfterValidator, model_validator, ConfigDict


def normalize_keywords(v: str) -> str:
    """标准化诊断关键词格式（小写化并去除首尾空格）

    Args:
        v: 原始诊断关键词字符串

    Returns:
        str: 标准化后的关键词

    Raises:
        ValueError: 如果输入不是字符串类型
    """
    if not isinstance(v, str):
        raise ValueError("诊断关键词必须是字符串类型")
    return v.lower().strip()


NormalizedStr = Annotated[
    str,
    AfterValidator(normalize_keywords),
    Field(description="经过标准化的诊断关键词（小写，去除空格）")
]


class EyeDiagnosisLabel(BaseModel):
    """眼底图像诊断数据标注模型

    本模型用于标准化眼底影像诊断数据的标注格式，确保数据的一致性和有效性。
    包含患者基本信息、影像路径、诊断关键词和结构化疾病标签体系。

    Attributes:
        id: 患者唯一标识符（整型，必须大于等于0）
        patient_age: 患者年龄（0-120岁，严格整型）
        patient_sex: 患者性别（Female/Male）
        left_fundus: 左眼影像文件路径（支持jpg/jpeg/png格式）
        right_fundus: 右眼影像文件路径（支持jpg/jpeg/png格式）
        left_diagnostic_keywords: 左侧诊断关键词（自动标准化为小写）
        right_diagnostic_keywords: 右侧诊断关键词（自动标准化为小写）
        n (N): 正常标签（True表示存在）
        d (D): 糖尿病视网膜病变标签
        g (G): 青光眼标签
        c (C): 白内障标签
        a (A): 年龄相关性黄斑变性标签
        h (H): 高血压视网膜病变标签
        m (M): 近视性视网膜病变标签
        o (O): 其他异常标签
    """
    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
        frozen=True,
        strict=True,
    )

    id: int = Field(..., description="患者唯一标识符", examples=[0])
    patient_age: Annotated[
        int,
        Field(
            ge=0,
            le=120,
            description="患者年龄（岁），范围：[0, 120]",
            examples=[69],
            strict=True
        )
    ]
    patient_sex: Annotated[
        Literal["Female", "Male"],
        Field(
            description="患者性别，允许值：['Female', 'Male']",
            examples=["Female"]
        )
    ]
    left_fundus: Annotated[
        str,
        Field(
            min_length=1,
            pattern=r"(?i)^.*\.(jpg|jpeg|png)$",  # 包含完整路径验证
            description="左眼底图像文件路径（相对路径）",
            examples=["0_left.jpg"],
            json_schema_extra={"format": "file-path"}
        )
    ]
    right_fundus: Annotated[
        str,
        Field(
            min_length=1,
            pattern=r"(?i)^.*\.(jpg|jpeg|png)$",
            description="右眼底图像文件路径（相对路径）",
            examples=["0_right.jpg"]
        )
    ]
    left_diagnostic_keywords: Annotated[
        NormalizedStr,
        Field(
            min_length=1,
            description="左侧诊断关键词（小写标准化）",
            examples=["cataract"]
        )
    ]
    right_diagnostic_keywords: Annotated[
        NormalizedStr,
        Field(
            min_length=1,
            description="右侧诊断关键词（小写标准化）",
            examples=["normal fundus"]
        )
    ]
    # 疾病标签字段（使用strict模式自动验证类型）
    n: Annotated[bool, Field(alias="N", description="正常 (Normal)", examples=[False])]
    d: Annotated[bool, Field(alias="D", description="糖尿病视网膜病变", examples=[False])]
    g: Annotated[bool, Field(alias="G", description="青光眼", examples=[False])]
    c: Annotated[bool, Field(alias="C", description="白内障", examples=[True])]
    a: Annotated[bool, Field(alias="A", description="年龄相关性黄斑变性", examples=[False])]
    h: Annotated[bool, Field(alias="H", description="高血压视网膜病变", examples=[False])]
    m: Annotated[bool, Field(alias="M", description="近视性视网膜病变", examples=[False])]
    o: Annotated[bool, Field(alias="O", description="其他异常", examples=[False])]

    # 模型级别验证
    @model_validator(mode="after")
    def check_at_least_one_disease(self) -> "EyeDiagnosisLabel":
        """疾病标签互斥性校验

        验证逻辑：
        - 确保至少有一个疾病标签为阳性（True）
        - 当所有疾病标签均为阴性时抛出验证错误

        Raises:
            ValueError: 当所有疾病标签均为阴性时

        Returns:
            EyeDiagnosisLabel: 校验通过的模型实例
        """
        disease_fields = ['n', 'd', 'g', 'c', 'a', 'h', 'm', 'o']
        if not any(getattr(self, field) for field in disease_fields):
            raise ValueError(
                "至少需要一个疾病标签为 True\n"
                "临床标注要求：正常标签(N)为False时必须至少存在一个异常标签"
            )
        return self
