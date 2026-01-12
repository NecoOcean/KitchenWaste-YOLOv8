# 模型文件目录

## 说明

此目录用于存放训练好的模型文件。模型文件需要从云服务器下载到本地。

## 下载方式

### 从 AutoDL 服务器下载

```bash
# 替换 <端口> 和 <IP> 为实际值
scp -P <端口> root@<IP>:/root/autodl-tmp/YOLOv8/runs/detect/*/weights/best.pt ./resources/models/
```

### 模型文件列表

| 文件名 | 说明 | 大小 |
|--------|------|------|
| `best.pt` | 最佳训练模型 | ~50-200MB |
| `last.pt` | 最后一个epoch的模型 | ~50-200MB |

## 使用方法

将下载的 `best.pt` 放置在此目录，程序会自动加载。

配置文件中的模型路径：
```python
# config/Config.py
model_path = os.path.join(PROJECT_ROOT, 'resources', 'models', 'best.pt')
```

## 注意事项

- 模型文件（*.pt）已在 `.gitignore` 中排除
- 请勿将模型文件提交到 Git 仓库
- 建议使用云存储备份重要的模型文件
