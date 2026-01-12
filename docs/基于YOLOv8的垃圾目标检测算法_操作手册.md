# 基于YOLOv8的垃圾目标检测算法 - 项目操作手册

## 目录
1. [项目概述](#1-项目概述)
2. [技术栈](#2-技术栈)
3. [数据集准备](#3-数据集准备)
4. [项目文件结构](#4-项目文件结构)
5. [环境配置](#5-环境配置)
6. [实现步骤](#6-实现步骤)
7. [功能说明](#7-功能说明)
8. [运行与测试](#8-运行与测试)

---

## 1. 项目概述

### 1.1 项目目标
构建一个基于 **YOLOv8** 深度学习目标检测算法的智能垃圾分类识别系统，应用于智能家居场景（厨房环境），实现垃圾的自动识别与分类指导。

### 1.2 应用场景
- **场景设定**：现代家庭厨房
- **硬件形态**：平板电脑/触摸屏 + 摄像头 + 语音模块
- **使用时机**：烹饪过程中和餐后清理时

### 1.3 使用流程
```
用户准备丢垃圾 → 将垃圾放到摄像头前 → 系统自动识别 → 显示分类结果 
→ 语音/文字指导 → 用户正确投放 → 记录统计
```

### 1.4 研究意义
- **理论意义**：探索深度学习在家居环境垃圾识别中的应用
- **实际意义**：提高家庭垃圾分类的准确性和效率，推动垃圾分类政策有效实施

---

## 2. 技术栈

### 2.1 核心技术

| 技术领域 | 使用技术 | 版本建议 | 说明 |
|---------|---------|---------|------|
| **深度学习框架** | Ultralytics YOLO | YOLOv8 | 目标检测模型 |
| **GUI框架** | PyQt5 | 5.15+ | 跨平台桌面应用 |
| **图像处理** | OpenCV | 4.5+ | 图像/视频处理 |
| **数值计算** | NumPy | 1.21+ | 数组矩阵运算 |
| **深度学习后端** | PyTorch | 1.8+ | 神经网络计算 |
| **编程语言** | Python | 3.8+ | 主开发语言 |

### 2.2 依赖清单 (requirements.txt)
```
PyQt5>=5.15.0
opencv-python>=4.5.0
ultralytics>=8.0.0
numpy>=1.21.0
torch>=1.8.0
torchvision>=0.9.0
pillow>=8.0.0
pyyaml>=5.4.0
```

### 2.3 YOLOv8 技术特点
- **单阶段检测**：速度快，适合实时检测
- **高精度**：优秀的检测准确率
- **多尺度检测**：有效检测不同大小的目标
- **易于训练**：Ultralytics提供完善的训练接口

---

## 3. 数据集准备

### 3.1 垃圾分类类别

#### 3.1.1 厨余垃圾（湿垃圾）
| 子类 | 具体物品 |
|-----|---------|
| 蔬菜类 | 菜叶、菜根、果皮、果核 |
| 肉类 | 骨头、肉皮、内脏 |
| 主食类 | 米饭、面条、面包屑 |
| 其他 | 茶叶渣、咖啡渣、蛋壳 |

#### 3.1.2 可回收垃圾（干垃圾）
| 子类 | 具体物品 |
|-----|---------|
| 塑料 | 食品袋、保鲜膜、塑料盒 |
| 纸类 | 纸盒、包装纸、餐巾纸 |
| 金属 | 易拉罐、罐头盒、铝箔 |
| 玻璃 | 调料瓶、酒瓶 |

#### 3.1.3 有害垃圾
- 过期调料、药品
- 清洁剂容器
- 电池（厨房电子秤等）

### 3.2 数据集结构（整合后的厨房垃圾分类数据集）
```
datasets/
├── kitchen_garbage/        # 整合后的厨房垃圾分类数据集
│   ├── images/
│   │   ├── train/          # 训练集图片（19,028张）
│   │   └── val/            # 验证集图片（18,653张）
│   ├── labels/
│   │   ├── train/          # 训练集标注（54,609个标注）
│   │   └── val/            # 验证集标注（60,074个标注）
│   └── data.yaml           # 数据集配置文件
├── images/                 # 原始数据集图片
│   ├── train/
│   └── val/
├── labels/                 # 原始数据集标注
│   ├── train/
│   └── val/
├── videos/                 # 测试视频
│   └── Cigrette.MP4
└── data.yaml               # 原始数据集配置
```

### 3.3 当前使用的类别配置（5类别）

基于现有数据集整合，当前项目使用以下5个类别：

| ID | 英文名 | 中文名 | 垃圾分类 | 标注数量 |
|----|--------|--------|----------|----------|
| 0 | fruit_peel | 果皮 | 厨余垃圾 | 5,324 |
| 1 | tea_leaves | 茶叶渣 | 厨余垃圾 | 821 |
| 2 | zip_top_can | 易拉罐 | 可回收物 | 1,557 |
| 3 | expired_medicine | 过期药品 | 有害垃圾 | 1,069 |
| 4 | other_garbage | 其他垃圾 | 其他垃圾 | 45,838 |

### 3.4 data.yaml 配置文件（厨房垃圾分类）

当前数据集配置为 **5个类别**（class_id: 0-4），基于现有数据集整合：

```yaml
# 厨房垃圾分类数据集配置
path: datasets/kitchen_garbage
train: images/train
val: images/val

# 类别数量 - 基于现有数据集整合（5个类别）
nc: 5

# 类别名称映射
names:
  0: fruit_peel        # 果皮（厨余垃圾）
  1: tea_leaves        # 茶叶渣（厨余垃圾）
  2: zip_top_can       # 易拉罐（可回收物）
  3: expired_medicine  # 过期药品（有害垃圾）
  4: other_garbage     # 其他垃圾
```

> **说明**：数据集已通过 `prepare_dataset.py` 脚本自动整合，将原始40类别数据集转换为5类别厨房垃圾分类数据集。

### 3.5 YOLO标注格式
每个txt标注文件格式：
```
<class_id> <x_center> <y_center> <width> <height>
```
- 所有坐标均为归一化值（0-1之间）
- 一行对应一个目标

示例（img_001.txt）：
```
0 0.5 0.5 0.3 0.4
4 0.2 0.3 0.15 0.2
```

### 3.5 数据采集建议
- **图片数量**：每类至少500-1000张图片
- **拍摄角度**：多角度（俯视、侧视、斜视）
- **光照条件**：不同光照（自然光、灯光、暗光）
- **背景环境**：真实厨房环境背景
- **标注工具**：推荐使用 LabelImg、Roboflow、CVAT

---

## 4. 项目文件结构

### 4.1 完整目录结构
```
YOLOv8_GarbageDetection/
│
├── main.py                     # 主应用程序入口
├── detection_service.py        # 检测服务模块
├── ui_manager.py               # UI管理模块
├── file_handler.py             # 文件处理模块
├── statistics_manager.py       # 统计管理模块
├── Config.py                   # 配置文件
├── detect_tools.py             # 工具函数
├── train.py                    # 模型训练脚本
├── requirements.txt            # 依赖列表
├── README.md                   # 项目说明
│
├── UIProgram/                  # UI界面模块
│   ├── __init__.py
│   ├── UiMain.py               # 主窗口UI定义（由Qt Designer生成）
│   ├── QssLoader.py            # QSS样式加载器
│   ├── precess_bar.py          # 进度条组件
│   └── style.css               # 界面样式表
│
├── models/                     # 模型文件目录
│   ├── yolov8n.pt              # 预训练模型
│   └── best.pt                 # 训练后最佳模型
│
├── datasets/                   # 数据集目录
│   ├── kitchen_garbage/        # 整合后的厨房垃圾分类数据集（当前使用）
│   │   ├── images/train/       # 训练集图片（19,028张）
│   │   ├── images/val/         # 验证集图片（18,653张）
│   │   ├── labels/train/       # 训练集标注
│   │   ├── labels/val/         # 验证集标注
│   │   └── data.yaml           # 数据集配置（5类别）
│   ├── images/                 # 原始数据集图片
│   ├── labels/                 # 原始数据集标注
│   └── data.yaml               # 原始数据集配置
│
├── runs/                       # 训练输出目录
│   └── detect/
│       └── kitchen_garbage_5cls/
│           ├── weights/
│           │   ├── best.pt     # 最佳模型
│           │   └── last.pt     # 最后一轮模型
│           └── results.csv
│
├── save_data/                  # 检测结果保存目录
├── TestFiles/                  # 测试文件目录
│   ├── images/
│   └── videos/
│
└── Font/                       # 字体文件目录
```

### 4.2 核心模块功能说明

| 文件 | 功能描述 |
|-----|---------|
| `main.py` | 应用程序入口，主窗口控制，事件处理，状态管理 |
| `detection_service.py` | YOLO模型加载，推理执行，结果封装 |
| `ui_manager.py` | 界面更新逻辑，图像显示，表格管理 |
| `file_handler.py` | 文件类型判断，路径生成，目录操作 |
| `statistics_manager.py` | 检测记录统计，数据持久化，CSV导出 |
| `Config.py` | 全局配置（模型路径、类别名称、保存路径等） |
| `train.py` | 模型训练脚本 |

---

## 5. 环境配置

### 5.1 安装Python环境
```bash
# 推荐使用Anaconda创建虚拟环境
conda create -n garbage_detect python=3.9
conda activate garbage_detect
```

### 5.2 安装PyTorch（GPU版本）
```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 或 CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 5.3 安装项目依赖
```bash
pip install -r requirements.txt
```

### 5.4 验证安装
```python
# 验证PyTorch和CUDA
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# 验证Ultralytics
from ultralytics import YOLO
print("Ultralytics安装成功")
```

---

## 6. 实现步骤

### 6.1 步骤总览

```
步骤1: 数据集准备与标注
    ↓
步骤2: 创建项目结构
    ↓
步骤3: 配置文件编写
    ↓
步骤4: 模型训练
    ↓
步骤5: 检测服务开发
    ↓
步骤6: UI界面开发
    ↓
步骤7: 系统集成
    ↓
步骤8: 测试与优化
```

### 6.2 步骤1：数据集准备与标注

**1. 收集图片**
- 在厨房环境拍摄各类垃圾图片
- 使用公开数据集补充（如TACO、TrashNet等）

**2. 标注工具安装**
```bash
pip install labelImg
labelImg  # 启动标注工具
```

**3. 标注要求**
- 使用YOLO格式
- 确保边界框紧贴目标
- 每张图片标注所有可见目标

### 6.3 步骤2：创建Config.py配置文件

```python
# Config.py - 厨房垃圾分类配置（5类别）
# -*- coding: utf-8 -*-

# 图片及视频检测结果保存路径
save_path = 'save_data'

# 使用的模型路径
model_path = 'models/best.pt'

# 类别数量
NUM_CLASSES = 5

# 类别配置（基于现有数据集整合）
names = {
    0: 'fruit_peel',        # 果皮
    1: 'tea_leaves',        # 茶叶渣
    2: 'zip_top_can',       # 易拉罐
    3: 'expired_medicine',  # 过期药品
    4: 'other_garbage'      # 其他垃圾
}

# 中文类别名称
CH_names = ['果皮', '茶叶渣', '易拉罐', '过期药品', '其他垃圾']

# 垃圾分类指导映射
classification_guide = {
    0: {'category': '厨余垃圾', 'color': 'green', 'tip': '请投入绿色厨余垃圾桶'},
    1: {'category': '厨余垃圾', 'color': 'green', 'tip': '请投入绿色厨余垃圾桶'},
    2: {'category': '可回收物', 'color': 'blue', 'tip': '请清洗后投入蓝色可回收垃圾桶'},
    3: {'category': '有害垃圾', 'color': 'red', 'tip': '请投入红色有害垃圾桶'},
    4: {'category': '其他垃圾', 'color': 'gray', 'tip': '请投入灰色其他垃圾桶'}
}
```

### 6.4 步骤3：模型训练脚本（厨房垃圾分类）

```python
# train.py
# coding:utf-8
from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    # 加载预训练模型
    model = YOLO('yolov8n.pt')  # 或使用 yolov8s.pt/yolov8m.pt
    
    # 训练配置 - 厨房垃圾分类（5个类别）
    results = model.train(
        data='datasets/kitchen_garbage/data.yaml',  # 整合后的数据集配置
        epochs=100,                              # 训练轮次
        imgsz=640,                               # 输入图像尺寸
        batch=16,                                # 批次大小（根据显存调整）
        cos_lr=True,                             # 余弦学习率调度
        optimizer='Adam',                        # 优化器
        device='0',                              # GPU设备，无GPU使用'cpu'
        patience=20,                             # 早停耐心值
        save=True,                               # 保存模型
        project='runs/detect',                   # 输出目录
        name='kitchen_garbage_5cls'              # 实验名称 - 厨房垃圾分类(5类)
    )
    
    # 验证模型
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")
    
    # 训练完成后，将best.pt复制到models目录
    # cp runs/detect/kitchen_garbage_5cls/weights/best.pt models/best.pt
```

### 6.5 步骤4：检测服务模块

```python
# detection_service.py
# -*- coding: utf-8 -*-
import time
import numpy as np
from ultralytics import YOLO
from typing import List
import Config


class DetectionResult:
    """检测结果数据类"""
    def __init__(self, results, elapsed_time: float):
        self.raw_results = results
        self.elapsed_time = elapsed_time
        self.locations = []
        self.classes = []
        self.confidences = []
        self.confidence_strings = []
        self._parse_results()
    
    def _parse_results(self):
        """解析YOLO检测结果"""
        if self.raw_results.boxes:
            location_list = self.raw_results.boxes.xyxy.tolist()
            self.locations = [list(map(int, bbox)) for bbox in location_list]
            cls_list = self.raw_results.boxes.cls.tolist()
            self.classes = [int(cls) for cls in cls_list]
            conf_list = self.raw_results.boxes.conf.tolist()
            self.confidences = conf_list
            self.confidence_strings = [f'{conf * 100:.2f} %' for conf in conf_list]
    
    @property
    def count(self) -> int:
        return len(self.classes)
    
    @property
    def has_detections(self) -> bool:
        return self.count > 0
    
    def get_plotted_image(self) -> np.ndarray:
        return self.raw_results.plot()
    
    def get_classification_guide(self) -> List[dict]:
        """获取分类指导信息"""
        guides = []
        for cls_id in self.classes:
            guide = Config.classification_guide.get(cls_id, {})
            guides.append({
                'name': Config.CH_names[cls_id],
                'category': guide.get('category', '未知'),
                'tip': guide.get('tip', '请查阅分类指南')
            })
        return guides


class DetectionService:
    """目标检测服务类"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        try:
            self.model = YOLO(self.model_path, task='detect')
            self.model(np.zeros((48, 48, 3)))  # 预热
            print(f"[INFO] 模型加载成功: {self.model_path}")
        except Exception as e:
            print(f"[ERROR] 模型加载失败: {e}")
            raise
    
    def detect(self, source) -> DetectionResult:
        start_time = time.time()
        results = self.model(source)[0]
        elapsed_time = time.time() - start_time
        return DetectionResult(results, elapsed_time)
```

### 6.6 步骤5：UI界面开发

使用 **Qt Designer** 设计界面，或直接编写PyQt5代码。

主要UI组件：
- 图像显示区域 (QLabel)
- 检测结果表格 (QTableWidget)
- 功能按钮（打开图片、视频、摄像头、保存）
- 检测信息显示（类别、置信度、坐标）
- 分类指导显示区域

### 6.7 步骤6：主程序集成

```python
# main.py 核心结构
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # 初始化核心组件
        self.detection_service = DetectionService(Config.model_path)
        self.ui_manager = UIManager(self.ui)
        
        # 连接信号槽
        self._connect_signals()
    
    def _connect_signals(self):
        self.ui.PicBtn.clicked.connect(self.on_open_image)
        self.ui.VideoBtn.clicked.connect(self.on_open_video)
        self.ui.CapBtn.clicked.connect(self.on_toggle_camera)
        self.ui.SaveBtn.clicked.connect(self.on_save)
    
    def on_open_image(self):
        # 打开图片 → 检测 → 显示结果
        file_path, _ = QFileDialog.getOpenFileName(...)
        result = self.detection_service.detect(file_path)
        self.ui_manager.update_detection_display(result, file_path)
        
        # 显示分类指导
        guides = result.get_classification_guide()
        self.ui_manager.show_classification_guide(guides)
```

---

## 7. 功能说明

### 7.1 核心功能
| 功能 | 描述 |
|-----|------|
| 图片检测 | 单张图片垃圾识别 |
| 批量检测 | 文件夹批量处理 |
| 视频检测 | 视频文件逐帧检测 |
| 摄像头检测 | 实时画面检测 |
| 分类指导 | 显示垃圾分类建议 |
| 结果保存 | 保存检测结果图片/视频 |
| 检测统计 | 记录并统计检测历史数据 |
| 数据导出 | 导出统计数据为CSV报表 |

### 7.2 检测输出信息
- 目标类别（中英文）
- 置信度百分比
- 边界框坐标
- 检测耗时
- 分类类别（厨余/可回收/有害/其他）
- 投放指导建议

### 7.3 检测统计功能

#### 7.3.1 功能概述
系统自动记录每次检测结果，并提供统计分析功能，帮助用户了解垃圾分类情况。

#### 7.3.2 统计面板
界面右侧显示统计面板，包含：
- **今日检测**：显示今日检测次数和检测项目总数
- **分类统计**：按四大垃圾类别显示统计数量
  - 🟢 厨余垃圾（绿色）
  - 🔵 可回收物（蓝色）
  - 🔴 有害垃圾（红色）
  - ⚫ 其他垃圾（灰色）

#### 7.3.3 数据存储
统计数据自动保存在 `save_data/statistics.json` 文件中，格式如下：
```json
{
  "id": 1,
  "timestamp": "2026-01-06T23:30:00",
  "date": "2026-01-06",
  "time": "23:30:00",
  "total_count": 3,
  "items": [
    {"class_id": 8, "name": "果皮", "category": "厨余垃圾", "confidence": 0.95},
    {"class_id": 23, "name": "易拉罐", "category": "可回收物", "confidence": 0.88}
  ],
  "elapsed_time": 0.045
}
```

#### 7.3.4 统计操作
| 操作 | 说明 |
|-----|------|
| 导出统计 | 点击"📊 导出统计"按钮，将数据导出为CSV文件 |
| 清空记录 | 点击"🗑 清空记录"按钮，清除所有历史统计（需确认） |

#### 7.3.5 CSV导出格式
导出的CSV文件包含以下字段：
```
记录ID,日期,时间,检测数量,类别详情,垃圾分类,检测耗时(ms)
1,2026-01-06,23:30:00,2,果皮;易拉罐,厨余垃圾;可回收物,45.2
```

#### 7.3.6 统计管理模块API
```python
from statistics_manager import StatisticsManager

# 初始化
stats = StatisticsManager()

# 获取今日统计
today = stats.get_today_statistics()
# 返回: {'date': '2026-01-06', 'detection_count': 10, 'total_items': 25, 'category_breakdown': {...}}

# 获取分类统计
category = stats.get_category_statistics()
# 返回: {'厨余垃圾': 10, '可回收物': 8, '有害垃圾': 2, '其他垃圾': 5}

# 获取类别统计
classes = stats.get_class_statistics()
# 返回: {'果皮': 5, '易拉罐': 3, '药品': 2, ...}

# 导出CSV
export_path = stats.export_to_csv()

# 清空记录
stats.clear_records()
```

---

## 8. 运行与测试

### 8.1 训练模型
```bash
python train.py
```

### 8.2 运行应用程序
```bash
python main.py
```

### 8.3 测试检测效果
```python
# 快速测试脚本
from ultralytics import YOLO

model = YOLO('models/best.pt')
results = model('TestFiles/images/test.jpg')
results[0].show()  # 显示结果
```

### 8.4 性能优化建议
- **模型选择**：轻量设备用yolov8n，高性能设备用yolov8s/m
- **输入尺寸**：根据目标大小调整imgsz（320/416/640）
- **批处理**：GPU推理时增大batch提升吞吐量
- **半精度推理**：使用FP16加速

---

## 附录

### A. 常见问题

**Q1: CUDA内存不足**
```python
# 减小batch size
results = model.train(batch=8)  # 或更小
```

**Q2: 检测精度低**
- 增加训练数据
- 延长训练epochs
- 调整学习率

**Q3: 检测速度慢**
- 使用更轻量的模型（yolov8n）
- 减小输入图像尺寸
- 启用GPU加速

### B. 参考资源
- [Ultralytics YOLOv8 文档](https://docs.ultralytics.com/)
- [PyQt5 官方文档](https://www.riverbankcomputing.com/static/Docs/PyQt5/)
- [OpenCV Python 教程](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

---

*操作手册生成时间：2026年1月6日*
