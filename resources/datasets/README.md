# 数据集目录

## 说明

本项目数据集存储在阿里云盘，不包含在代码仓库中。

## 下载地址

**阿里云盘**: [请在此处添加分享链接]

## 数据集结构

下载后解压到此目录，确保结构如下：

```
resources/datasets/
├── images/
│   ├── train/    # 训练集图片 (~16,840 张)
│   └── val/      # 验证集图片 (~1,776 张)
├── labels/
│   ├── train/    # 训练集标签
│   └── val/      # 验证集标签
├── data.yaml     # 数据集配置文件（预处理后生成）
└── processed/    # 预处理后的数据集（运行脚本后生成）
```

## 使用步骤

1. **下载数据集**
   - 从阿里云盘下载压缩包
   - 解压到 `resources/datasets/` 目录

2. **运行预处理**
   ```bash
   python tools/dataset_preprocessor.py --execute
   ```

3. **验证数据集**
   ```bash
   # 检查文件数量
   python -c "from pathlib import Path; print(f'Train images: {len(list((Path(\"resources/datasets/images/train\")).glob(\"*\")))}'); print(f'Val images: {len(list((Path(\"resources/datasets/images/val\")).glob(\"*\")))}')"
   ```

## 数据集统计

| 类别 | 训练集 | 验证集 |
|------|--------|--------|
| 图片 | ~16,840 | ~1,776 |
| 标签 | ~16,840 | ~1,776 |
| 类别数 | 40 | 40 |

## 注意事项

- 数据集文件已在 `.gitignore` 中排除
- 请勿将数据集文件提交到 Git 仓库
- 大文件会显著增加仓库体积
