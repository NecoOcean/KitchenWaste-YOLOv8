# KitchenWaste-YOLOv8

**基于YOLOv8的智能厨房垃圾分类识别系统**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-1.8+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg" alt="YOLOv8">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  <img src="https://img.shields.io/github/stars/NecoOcean/KitchenWaste-YOLOv8?style=social" alt="Stars">
</p>

## 📖 项目简介

本项目是一个基于 **YOLOv8** 深度学习目标检测算法的智能垃圾分类识别系统，应用于智能家居场景（厨房环境），实现垃圾的自动识别与分类指导。

### 🎯 应用场景

- **场景设定**：现代家庭厨房
- **硬件形态**：平板电脑/触摸屏 + 摄像头
- **使用流程**：用户将垃圾放到摄像头前 → 系统自动识别 → 显示分类结果和投放指导

## ✨ 功能特点

| 功能 | 描述 |
|------|------|
| 📷 **图片检测** | 单张图片垃圾识别 |
| 📁 **批量检测** | 文件夹批量处理 |
| 🎬 **视频检测** | 视频文件逐帧检测 |
| 📹 **摄像头检测** | 实时画面检测 |
| 🏷️ **分类指导** | 显示垃圾分类建议（厨余/可回收/有害/其他） |
| 💾 **结果保存** | 保存检测结果图片/视频 |
| 📊 **检测统计** | 记录并统计检测历史数据 |
| 📋 **数据导出** | 导出统计数据为CSV报表 |

## 🛠️ 技术栈

| 技术 | 版本 | 说明 |
|------|------|------|
| **Ultralytics YOLO** | YOLOv8 | 目标检测模型 |
| **PyQt5** | 5.15+ | 桌面GUI框架 |
| **OpenCV** | 4.5+ | 图像/视频处理 |
| **PyTorch** | 1.8+ | 深度学习后端 |
| **Python** | 3.8+ | 主开发语言 |

## 📂 项目结构

```
YOLOv8/
│
├── 🟢 core/                        # 核心应用层
│   ├── main.py                     # 主程序入口
│   ├── detection_service.py        # 检测服务模块
│   ├── ui_manager.py               # UI管理模块
│   ├── file_handler.py             # 文件处理模块
│   └── statistics_manager.py       # 统计管理模块
│
├── 🔵 UIProgram/                   # UI界面层
│   ├── UiMain.py                   # 主窗口UI定义
│   ├── QssLoader.py                # 样式加载器
│   ├── style.css                   # 界面样式表
│   └── __init__.py
│
├── 🟡 config/                      # 配置与训练
│   ├── Config.py                   # 运行配置
│   ├── Config_kitchen.py           # 厨房场景配置
│   └── train.py                    # 模型训练脚本
│
├── 🟠 tools/                       # 数据处理工具
│   ├── dataset_preprocessor.py     # 数据预处理
│   └── convert_labels.py           # 标签转换
│
├── 📂 resources/                   # 资源目录
│   ├── datasets/                   # 数据集（需从云盘下载）
│   │   ├── images/train/           # 训练集图片
│   │   ├── images/val/             # 验证集图片
│   │   ├── labels/train/           # 训练集标签
│   │   └── labels/val/             # 验证集标签
│   ├── models/                     # 模型文件（需训练或下载）
│   ├── save_data/                  # 检测结果保存
│   ├── TestFiles/                  # 测试文件
│   └── Font/                       # 字体文件
│
├── 📄 docs/                        # 文档
│   ├── 垃圾分类项目要求.md
│   ├── 基于YOLOv8的垃圾目标检测算法.md
│   ├── 基于YOLOv8的垃圾目标检测算法_操作手册.md
│   └── 数据集预处理操作指南.md
│
├── run.py                          # 启动脚本
├── requirements.txt                # 依赖列表
├── .gitignore                      # Git忽略配置
└── README.md                       # 项目说明
```

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/NecoOcean/KitchenWaste-YOLOv8.git
cd KitchenWaste-YOLOv8
```

### 2. 创建虚拟环境

```bash
conda create -n garbage_detect python=3.9
conda activate garbage_detect
```

### 3. 安装PyTorch（GPU版本）

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 或 CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. 安装项目依赖

```bash
pip install -r requirements.txt
```

### 5. 准备数据集

数据集存储在阿里云盘，请从以下链接下载：

> **阿里云盘**: [下载链接]（请联系作者获取）

下载后解压到 `resources/datasets/` 目录。

### 6. 数据预处理

```bash
# 模拟运行（预览操作）
python tools/dataset_preprocessor.py

# 执行实际预处理
python tools/dataset_preprocessor.py --execute
```

### 7. 训练模型

```bash
python config/train.py
```

训练完成后，将 `runs/detect/*/weights/best.pt` 复制到 `resources/models/` 目录。

### 8. 运行应用程序

```bash
python run.py
```

## ⌨️ 快捷键

| 快捷键 | 功能 |
|--------|------|
| `Ctrl+O` | 打开图片 |
| `Ctrl+Q` | 退出程序 |
| `←/→` | 切换图片（批量模式） |
| `Esc` | 停止视频/摄像头 |

## 📊 支持的垃圾类别

系统支持 **29个** 垃圾类别，涵盖四大分类：

| 分类 | 颜色 | 示例 |
|------|------|------|
| 🟢 **厨余垃圾** | 绿色 | 菜叶、果皮、骨头、米饭、茶叶渣 |
| 🔵 **可回收物** | 蓝色 | 易拉罐、塑料瓶、纸盒、玻璃瓶 |
| 🔴 **有害垃圾** | 红色 | 过期药品、电池、清洁剂容器 |
| ⚫ **其他垃圾** | 灰色 | 餐巾纸、烟蒂、陶瓷碎片 |

## 📚 详细文档

- [操作手册](docs/基于YOLOv8的垃圾目标检测算法_操作手册.md) - 完整使用指南
- [数据预处理指南](docs/数据集预处理操作指南.md) - 数据集处理步骤
- [项目需求](docs/垃圾分类项目要求.md) - 需求定义

## 🖥️ 云服务器部署

支持在 AutoDL 等云平台部署训练，详见 [操作手册](docs/基于YOLOv8的垃圾目标检测算法_操作手册.md)。

## 📄 许可证

本项目采用 MIT License 开源协议。

## 🙏 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/)
- [OpenCV](https://opencv.org/)
