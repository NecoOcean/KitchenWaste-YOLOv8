# KitchenWaste-YOLOv8 数据集

## 说明

本项目数据集存储在阿里云盘，不包含在代码仓库中。

- **项目地址**：https://github.com/NecoOcean/KitchenWaste-YOLOv8
- **数据集名称**：KitchenWaste-YOLOv8-Dataset
- **类别数量**：29类

## 下载地址

> **阿里云盘**: [下载链接]（请联系作者获取）

---

## 方法一：本地电脑下载

1. 点击上方阿里云盘链接
2. 下载 `KitchenWaste-YOLOv8-Dataset.zip`
3. 解压到 `resources/datasets/` 目录

---

## 方法二：AutoDL 云服务器下载（推荐）

在 AutoDL 或其他 Linux 云服务器上使用 **aliyunpan-cli** 下载：

### 步骤 1：安装 aliyunpan 工具

```bash
cd /root
# 下载最新版aliyunpan
wget https://gitee.com/cumtsgw/aliyunpan/releases/download/v0.3.10/aliyunpan-linux-amd64.tar.gz

# 解压
tar -zxvf aliyunpan-linux-amd64.tar.gz

# 移动到系统路径（方便全局使用）
mv aliyunpan-linux-amd64/aliyunpan /usr/local/bin/

# 验证安装
aliyunpan --version
```

### 步骤 2：登录阿里云盘

```bash
# 执行登录命令
aliyunpan login

# 根据提示选择登录方式：
# 1. 扫码登录（推荐）
# 2. 手机号登录
# 3. RefreshToken登录
```

**获取 RefreshToken 方法**（如需）：
1. 浏览器打开阿里云盘网页版并登录
2. 打开开发者工具（F12）
3. 在 Application → Local Storage → token 中找到 refresh_token

### 步骤 3：查看云盘文件

```bash
# 列出云盘根目录
aliyunpan ls /

# 列出数据集目录（根据实际路径调整）
aliyunpan ls /备份文件/WorkData/Datasets/
```

### 步骤 4：下载数据集

```bash
# 下载到 autodl-tmp 目录（AutoDL 数据盘）
aliyunpan download /备份文件/WorkData/Datasets/KitchenWaste-YOLOv8-Dataset /root/autodl-tmp/

# 等待下载完成...
```

### 步骤 5：解压并移动到项目目录

```bash
# 进入下载目录
cd /root/autodl-tmp/KitchenWaste-YOLOv8-Dataset

# 如果是压缩包，先解压
# unzip KitchenWaste-YOLOv8-Dataset.zip

# 移动到项目目录
mv images labels data.yaml /root/KitchenWaste-YOLOv8/resources/datasets/
```

### 步骤 6：验证下载

```bash
# 检查文件结构
ls /root/KitchenWaste-YOLOv8/resources/datasets/
# 应显示: images/  labels/  data.yaml

# 检查图片数量
ls /root/KitchenWaste-YOLOv8/resources/datasets/images/train | wc -l
ls /root/KitchenWaste-YOLOv8/resources/datasets/images/val | wc -l
```

---

## 数据集结构

下载后确保目录结构如下：

```
resources/datasets/
├── images/
│   ├── train/    # 训练集图片
│   └── val/      # 验证集图片
├── labels/
│   ├── train/    # 训练集标签
│   └── val/      # 验证集标签
└── data.yaml     # 数据集配置文件
```

---

## 下载后步骤

### 1. 运行数据预处理（如需要）

```bash
cd /root/KitchenWaste-YOLOv8
python tools/dataset_preprocessor.py --execute
```

### 2. 验证数据集

```bash
python -c "
from pathlib import Path
train_imgs = len(list(Path('resources/datasets/images/train').glob('*')))
val_imgs = len(list(Path('resources/datasets/images/val').glob('*')))
print(f'训练集图片: {train_imgs}')
print(f'验证集图片: {val_imgs}')
"
```

### 3. 开始训练

```bash
python config/train.py
```

---

## 数据集统计

| 分类 | 类别ID | 数量 |
|------|--------|------|
| 🟢 厨余垃圾 | 0-12 | 13类 |
| 🔵 可回收物 | 13-23 | 11类 |
| 🔴 有害垃圾 | 24-27 | 4类 |
| ⚫ 其他垃圾 | 28 | 1类 |
| **总计** | - | **29类** |

---

## 常见问题

### Q1: aliyunpan 下载速度慢

```bash
# 设置多线程下载
aliyunpan config set -max_download_parallel 5
```

### Q2: 登录失效

```bash
# 重新登录
aliyunpan logout
aliyunpan login
```

### Q3: 磁盘空间不足

```bash
# 检查磁盘空间
df -h

# AutoDL 数据盘路径（大容量）
/root/autodl-tmp/
```

---

## 注意事项

- ⚠️ 数据集文件已在 `.gitignore` 中排除
- ⚠️ 请勿将数据集文件提交到 Git 仓库
- ⚠️ 大文件会显著增加仓库体积
- ✅ 建议将数据集下载到 AutoDL 数据盘 (`/root/autodl-tmp/`)
