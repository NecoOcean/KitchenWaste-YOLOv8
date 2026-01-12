# Config.py
# -*- coding: utf-8 -*-
"""
基于YOLOv8的垃圾目标检测算法 - 配置文件
基于现有数据集整理的5类别配置
"""

# 图片及视频检测结果保存路径
save_path = 'save_data'

# 使用的模型路径
model_path = 'models/best.pt'

# 类别数量
NUM_CLASSES = 5

# 类别配置（基于现有数据集）
names = {0: 'fruit_peel', 1: 'tea_leaves', 2: 'zip_top_can', 3: 'expired_medicine', 4: 'other_garbage'}

# 中文类别名称
CH_names = ['果皮', '茶叶渣', '易拉罐', '过期药品', '其他垃圾']

# 垃圾分类指导映射
classification_guide = {0: {'category': '厨余垃圾', 'color': 'green', 'tip': '请投入绿色厨余垃圾桶'}, 1: {'category': '厨余垃圾', 'color': 'green', 'tip': '请投入绿色厨余垃圾桶'}, 2: {'category': '可回收物', 'color': 'blue', 'tip': '请清洗后投入蓝色可回收垃圾桶'}, 3: {'category': '有害垃圾', 'color': 'red', 'tip': '请投入红色有害垃圾桶'}, 4: {'category': '其他垃圾', 'color': 'gray', 'tip': '请投入灰色其他垃圾桶'}}
