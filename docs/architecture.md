# Ocular Insight 项目架构文档

## 项目概述
Ocular Insight 是一个用于医学图像处理的项目，专注于眼球图像的多标签分类和回归任务。项目使用 Python 和 PyTorch 进行开发。

## 目录结构
```
ocular-insight/
├── configs/                        # 配置文件
│
├── data/                           # 数据相关
│   ├── raw/                        # 原始数据集（建议只读）
│   ├── processed/                  # 预处理后的数据集
│   └── splits/                     # 数据集划分文件
│
├── docs/                           # 开发文档
│   └── architecture.md             # 项目架构文档
│
├── models/                         # 模型定义
│   └── __init__.py
│
├── src/                            # 核心源代码
│   ├── data/                       # 数据相关模块
│   │   ├── __init__.py
│   │   └── schemas/                # 数据模型定义
│   │       └── eye_diagnosis.py    # 眼底图像标注数据模型
│   │
│   ├── utils/                      # 工具函数
│   │
│   ├── engine/                     # 训练引擎
│   │
│   └── cli/                        # CLI相关
│       └── __init__.py
│
├── scripts/                        # 实用脚本
│   └── convert_excel_to_csv.py     # 将 Excel 转换为 CSV
│
├── runs/                           # 训练结果
│
├── tests/                          # 单元测试
│
├── requirements.txt                # 依赖库
├── README.md                       # 项目文档
└── .gitignore
```

## 主要模块说明
- **configs/**: 存放训练和模型的配置文件。
- **data/**: 存放原始数据、预处理后的数据以及数据集划分文件。
- **docs/**: 存放项目开发文档，包括架构设计、API 文档等。
- **models/**: 定义项目中使用到的模型。
- **src/**: 核心源代码，包括数据处理、工具函数、训练引擎和 CLI 工具。
- **scripts/**: 实用脚本，如数据预处理、训练和预测脚本。
- **runs/**: 存放训练结果，包括模型检查点和日志。
- **tests/**: 单元测试代码。
- **requirements.txt**: 项目依赖库列表。
- **README.md**: 项目概述和使用说明。

## 后续更新
- 随着项目的进展，此文档将动态更新，以反映最新的项目结构和设计决策。

## 修改提示
在修改此文件时，不必将新内容包裹在代码块(```)中，因为重复的代码块标记将导致格式化错误，直接输出新内容即可。