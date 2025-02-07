
import torch

# 导入量化相关的模块
import torch.ao.quantization as quantization
from torch.ao.quantization import MinMaxObserver, FakeQuantize

# 从 baseline.convnext 导入 LayerNorm, ConvNeXtBlock, ConvNeXtLayer 和 ConvNeXt 的定义
# 假设 baseline.convnext.py 文件在同一目录下或者可以被 Python 找到
from src.models.baseline.convnext import (
    ConvNeXt, # 导入原始的 ConvNeXt 基类
)


class QATConvNeXt(ConvNeXt):  # 继承 ConvNeXt
    """
    量化感知训练 (QAT) 版本的 ConvNeXt 模型。
    继承自 Baseline ConvNeXt 模型，并添加了 QAT 所需的 QuantStub 和 DeQuantStub。
    """

    def __init__(self, *args, **kwargs):
        """
        初始化 QATConvNeXt 模型。
        与 Baseline ConvNeXt 的初始化参数相同，*args 和 **kwargs 会传递给父类 ConvNeXt 的 __init__ 方法。
        """
        super().__init__(*args, **kwargs)
        # 1. 创建 QuantStub 和 DeQuantStub 实例
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        QATConvNeXt 模型的前向传播。
        在模型输入端添加 quant = QuantStub(), 模型输出端添加 dequant = DeQuantStub()。
        """
        # 2. 在模型 forward 方法中，显式地插入 QuantStub 和 DeQuantStub
        x = self.quant(x)  # 量化输入
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.norm_head(x.mean([-2, -1]))  # 全局平均池化 (N, C, H, W) -> (N, C)
        x = self.head(x)
        x = self.dequant(x)  # 反量化输出
        return x


def __test__():
    # 示例使用
    batch_size: int = 2
    channels: int = 3
    height: int = 224
    width: int = 224
    num_classes: int = 5  # 假设你的疾病类别数量为 5

    # 创建一个随机输入张量
    dummy_input: torch.Tensor = torch.randn(batch_size, channels, height, width)

    # 初始化 QATConvNeXt 模型
    qat_model: QATConvNeXt = QATConvNeXt(
        in_channels=channels, num_classes=num_classes, depths=[2, 2, 2, 2], dims=[64, 128, 256, 512]
    )

    # 修改：创建自定义 QConfig，显式指定 quant_min, quant_max 和 dtype
    qat_config = quantization.QConfig(
        activation=FakeQuantize.with_args(observer=MinMaxObserver, quant_min=0, quant_max=255, dtype=torch.quint8),  # 显式指定 dtype=torch.quint8 for activation
        weight=FakeQuantize.with_args(observer=MinMaxObserver, quant_min=-128, quant_max=127, dtype=torch.qint8)   # 显式指定 dtype=torch.qint8 for weight
    )
    qat_model.qconfig = qat_config

    # 准备模型进行量化感知训练 (插入 Observer 和 FakeQuantize 模块)
    prepared_model = quantization.prepare_qat(qat_model.train()) # 注意设置为 train 模式

    print("模型已准备好进行 QAT.")

    # 打印模型结构 (可选，用于查看 QuantStub 和 DeQuantStub 是否已插入)
    print(prepared_model)

    # 模型推理 (前向传播测试)
    output: torch.Tensor = prepared_model(dummy_input)

    # 打印输出形状和内容 (前 batch_size 个样本的预测概率)
    print(f"输出形状: {output.shape}")  # 应该是 [batch_size, num_classes]
    print(f"输出示例 (前 {batch_size} 个样本):\n{output[:batch_size]}")


if __name__ == "__main__":
    __test__()
