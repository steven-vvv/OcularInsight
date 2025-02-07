from typing import Union, List, Optional

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """支持通道优先格式的 LayerNorm 层。"""

    def __init__(
        self,
        normalized_shape: Union[int, List[int], torch.Size],
        eps: float = 1e-6,
        data_format: str = "channels_last",
    ) -> None:
        """
        初始化 LayerNorm 类。

        Args:
            normalized_shape: 输入的尺寸，从该尺寸的最后一个维度开始将被归一化。
            eps: 为了数值稳定性添加到分母的值。默认为 1e-6。
            data_format: "channels_last" (N, H, W, C) 或 "channels_first" (N, C, H, W)。默认为 "channels_last"。

        Raises:
            ValueError: 如果 `data_format` 不是 "channels_last" 或 "channels_first"。
        """
        super().__init__()
        self.data_format = data_format
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)

        self.weight: nn.Parameter = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias: nn.Parameter = nn.Parameter(torch.zeros(self.normalized_shape))
        self.eps: float = eps

        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(
                f"data_format 必须是 channels_last 或 channels_first, 而不是 {self.data_format}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        LayerNorm 的前向传播。

        Args:
            x: 输入张量，形状为 (N, C, H, W) 或 (N, H, W, C)，取决于 `data_format`。

        Returns:
            归一化后的张量，形状与输入相同。
        """
        if self.data_format == "channels_last":
            return nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u: torch.Tensor = x.mean(dim=1, keepdim=True)
            s: torch.Tensor = (x - u).pow(2).mean(dim=1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        else:
            # 理论上不会到达这里，因为初始化时已经做了 data_format 的检查
            raise ValueError("Invalid data_format")


class ConvNeXtBlock(nn.Module):
    """ConvNeXt 块 (Block)."""

    def __init__(
        self,
        dim: int,
        drop_path_ratio: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        """
        初始化 ConvNeXtBlock 类。

        Args:
            dim: 输入通道数。
            drop_path_ratio: Stochastic Depth 的 dropout 概率。默认为 0.0。
            layer_scale_init_value: Layer Scale 的初始值。默认为 1e-6。
        """
        super().__init__()
        self.dwconv: nn.Conv2d = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # 深度可分离卷积
        self.norm: LayerNorm = LayerNorm(dim, eps=1e-6)
        self.pwconv1: nn.Linear = nn.Linear(dim, 4 * dim)  # pointwise/1x1 卷积，扩展通道数
        self.act: nn.GELU = nn.GELU()
        self.pwconv2: nn.Linear = nn.Linear(4 * dim, dim)  # pointwise/1x1 卷积，投影回原通道数
        self.gamma: Optional[nn.Parameter] = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0 else None
        )
        self.drop_path: nn.Module = (
            nn.Dropout(drop_path_ratio) if drop_path_ratio > 0.0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ConvNeXtBlock 的前向传播。

        Args:
            x: 输入张量，形状为 (N, C, H, W)。

        Returns:
            经过 ConvNeXtBlock 处理后的张量，形状与输入相同。
        """
        input_tensor: torch.Tensor = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2).contiguous()  # (N, H, W, C) -> (N, C, H, W)
        x = self.drop_path(x) + input_tensor
        return x


class ConvNeXtLayer(nn.Module):
    """ConvNeXt 层 (Layer).

    一个 Layer (Stage) 包含 [可选的下采样层, 多个 ConvNeXtBlock]。
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        drop_path_rate: float,
        layer_scale_init_value: float = 1e-6,
        downsample: bool = True,
    ) -> None:
        """
        初始化 ConvNeXtLayer 类。

        Args:
            dim: 输入通道数。
            depth: 该 Layer 中 ConvNeXtBlock 的数量。
            drop_path_rate: 该 Layer 中所有 ConvNeXtBlock 的 Stochastic Depth 的总 dropout 概率。
            layer_scale_init_value: Layer Scale 的初始值。默认为 1e-6。
            downsample: 是否在该 Layer 的开始进行下采样。默认为 True。
        """
        super().__init__()
        self.downsample: bool = downsample
        if self.downsample:
            self.downsample_layer: nn.Sequential = nn.Sequential(
                LayerNorm(dim, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dim, 2 * dim, kernel_size=2, stride=2),
            )
            dim = 2 * dim  # 更新 dim，因为下采样后通道数翻倍

        self.blocks: nn.ModuleList = nn.ModuleList(
            [
                ConvNeXtBlock(dim=dim, drop_path_ratio=drop_path_rate, layer_scale_init_value=layer_scale_init_value)
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ConvNeXtLayer 的前向传播。

        Args:
            x: 输入张量，形状为 (N, C, H, W)。

        Returns:
            经过 ConvNeXtLayer 处理后的张量，形状为 (N, C, H, W)。
        """
        if self.downsample:
            x = self.downsample_layer(x)
        for block in self.blocks:
            x = block(x)
        return x


class ConvNeXt(nn.Module):
    """ConvNeXt 模型主类."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        depths: Optional[List[int]] = None,
        dims: Optional[List[int]] = None,
        drop_path_rate: float = 0.1,
        layer_scale_init_value: float = 1e-6,
        head_init_scale: float = 1.0,
    ) -> None:
        """
        初始化 ConvNeXt 模型。

        Args:
            in_channels: 输入图像的通道数。默认为 3 (RGB 图像)。
            num_classes: 分类类别数。默认为 1000 (ImageNet 默认类别数)。
            depths: 每个 stage (layer) 中 ConvNeXtBlock 的数量列表。默认为 [3, 3, 9, 3]。
            dims: 每个 stage 的通道数列表。默认为 [96, 192, 384, 768]。
            drop_path_rate: 所有 Blocks 的 Stochastic Depth 的总 dropout 概率。默认为 0.1。
            layer_scale_init_value: Layer Scale 的初始值。默认为 1e-6。
            head_init_scale: 分类头部的初始尺度。默认为 1.0。
        """
        super().__init__()
        if depths is None:
            depths = [3, 3, 9, 3]
        if dims is None:
            dims = [96, 192, 384, 768]

        self.stem: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )

        self.stages: nn.ModuleList = nn.ModuleList()  # 四个 layer (stage) 的卷积块
        current_dim: int = dims[0]
        for i in range(len(depths)):
            downsample = True if i > 0 else False
            stage = ConvNeXtLayer(
                dim=current_dim,
                depth=depths[i],
                drop_path_rate=drop_path_rate,
                layer_scale_init_value=layer_scale_init_value,
                downsample=downsample,
            )
            self.stages.append(stage)
            if downsample:  # 如果进行了下采样，更新 current_dim
                current_dim = current_dim * 2  # 简化：下采样时总是翻倍

        self.norm_head: LayerNorm = LayerNorm(current_dim, eps=1e-6)  # final norm layer
        self.head: nn.Linear = nn.Linear(current_dim, num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """初始化权重。"""
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:  # 某些 layers 可能没有 bias
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取特征。

        Args:
            x: 输入图像张量，形状为 (N, C, H, W)。

        Returns:
            提取的特征张量，形状为 (N, C)。
        """
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return self.norm_head(x.mean([-2, -1]))  # 全局平均池化 (N, C, H, W) -> (N, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        模型的前向传播。

        Args:
            x: 输入图像张量，形状为 (N, C, H, W)。

        Returns:
            模型输出，疾病概率预测张量，形状为 (N, num_classes)。
        """
        x = self.forward_features(x)
        x = self.head(x)
        return x

def __test__() -> None:
    # 示例使用
    batch_size: int = 2
    channels: int = 3
    height: int = 224
    width: int = 224
    num_classes: int = 5  # 假设你的疾病类别数量为 5

    # 创建一个随机输入张量
    dummy_input: torch.Tensor = torch.randn(batch_size, channels, height, width)

    # 初始化 ConvNeXt 模型，假设用于 5 分类任务，并修改 dims 和 depths 以适应你的需求
    model: ConvNeXt = ConvNeXt(
        in_channels=channels, num_classes=num_classes, depths=[2, 2, 2, 2], dims=[64, 128, 256, 512]
    )
    print(f"norm_head normalized_shape in main: {model.norm_head.normalized_shape}")  # 打印 norm_head 的 normalized_shape

    # 模型推理
    output: torch.Tensor = model(dummy_input)

    # 打印输出形状和内容 (前 batch_size 个样本的预测概率)
    print(f"输出形状: {output.shape}")  # 应该是 [batch_size, num_classes]
    print(f"输出示例 (前 {batch_size} 个样本):\n{output[:batch_size]}")

if __name__ == "__main__":
    __test__()
