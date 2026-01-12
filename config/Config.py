# Config.py
# -*- coding: utf-8 -*-
"""
基于YOLOv8的垃圾目标检测算法 - 配置文件
厨房垃圾分类专用（29个类别）
"""
import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 图片及视频检测结果保存路径
save_path = os.path.join(PROJECT_ROOT, 'resources', 'save_data')

# 使用的模型路径
model_path = os.path.join(PROJECT_ROOT, 'resources', 'models', 'best.pt')

# 类别数量
NUM_CLASSES = 29

# 类别配置（与data.yaml保持一致）
names = {
    # 厨余垃圾 - 蔬菜类
    0: 'vegetable_leaves',    # 菜叶
    1: 'vegetable_roots',     # 菜根
    2: 'fruit_peel',          # 果皮
    3: 'fruit_core',          # 果核
    # 厨余垃圾 - 肉类
    4: 'bone',                # 骨头
    5: 'meat_skin',           # 肉皮
    6: 'offal',               # 内脏
    # 厨余垃圾 - 主食类
    7: 'rice',                # 米饭
    8: 'noodles',             # 面条
    9: 'bread_crumbs',        # 面包屑
    # 厨余垃圾 - 其他
    10: 'tea_leaves',         # 茶叶渣
    11: 'coffee_grounds',     # 咖啡渣
    12: 'eggshell',           # 蛋壳
    # 可回收垃圾 - 塑料
    13: 'plastic_bag',        # 食品袋
    14: 'plastic_wrap',       # 保鲜膜
    15: 'plastic_container',  # 塑料盒
    # 可回收垃圾 - 纸类
    16: 'paper_box',          # 纸盒
    17: 'wrapping_paper',     # 包装纸
    18: 'tissue',             # 餐巾纸
    # 可回收垃圾 - 金属
    19: 'zip_top_can',        # 易拉罐
    20: 'tin_can',            # 罐头盒
    21: 'aluminum_foil',      # 铝箔
    # 可回收垃圾 - 玻璃
    22: 'seasoning_bottle',   # 调料瓶
    23: 'wine_bottle',        # 酒瓶
    # 有害垃圾
    24: 'expired_seasoning',  # 过期调料
    25: 'expired_medicine',   # 过期药品
    26: 'cleaner_container',  # 清洁剂容器
    27: 'battery',            # 电池
    # 其他垃圾
    28: 'other_garbage',      # 其他垃圾
}

# 中文类别名称
CH_names = [
    '菜叶', '菜根', '果皮', '果核',           # 0-3: 蔬菜类
    '骨头', '肉皮', '内脏',                   # 4-6: 肉类
    '米饭', '面条', '面包屑',                 # 7-9: 主食类
    '茶叶渣', '咖啡渣', '蛋壳',               # 10-12: 厨余其他
    '食品袋', '保鲜膜', '塑料盒',             # 13-15: 塑料
    '纸盒', '包装纸', '餐巾纸',               # 16-18: 纸类
    '易拉罐', '罐头盒', '铝箔',               # 19-21: 金属
    '调料瓶', '酒瓶',                         # 22-23: 玻璃
    '过期调料', '过期药品', '清洁剂容器', '电池',  # 24-27: 有害垃圾
    '其他垃圾',                               # 28: 其他垃圾
]

# 垃圾分类指导映射
classification_guide = {
    # 厨余垃圾（绿色）- ID 0-12
    0: {'category': '厨余垃圾', 'color': 'green', 'tip': '请投入绿色厨余垃圾桶'},
    1: {'category': '厨余垃圾', 'color': 'green', 'tip': '请投入绿色厨余垃圾桶'},
    2: {'category': '厨余垃圾', 'color': 'green', 'tip': '请投入绿色厨余垃圾桶'},
    3: {'category': '厨余垃圾', 'color': 'green', 'tip': '请投入绿色厨余垃圾桶'},
    4: {'category': '厨余垃圾', 'color': 'green', 'tip': '请投入绿色厨余垃圾桶'},
    5: {'category': '厨余垃圾', 'color': 'green', 'tip': '请投入绿色厨余垃圾桶'},
    6: {'category': '厨余垃圾', 'color': 'green', 'tip': '请投入绿色厨余垃圾桶'},
    7: {'category': '厨余垃圾', 'color': 'green', 'tip': '请投入绿色厨余垃圾桶'},
    8: {'category': '厨余垃圾', 'color': 'green', 'tip': '请投入绿色厨余垃圾桶'},
    9: {'category': '厨余垃圾', 'color': 'green', 'tip': '请投入绿色厨余垃圾桶'},
    10: {'category': '厨余垃圾', 'color': 'green', 'tip': '请投入绿色厨余垃圾桶'},
    11: {'category': '厨余垃圾', 'color': 'green', 'tip': '请投入绿色厨余垃圾桶'},
    12: {'category': '厨余垃圾', 'color': 'green', 'tip': '请投入绿色厨余垃圾桶'},
    # 可回收垃圾（蓝色）- ID 13-23
    13: {'category': '可回收物', 'color': 'blue', 'tip': '请投入蓝色可回收垃圾桶'},
    14: {'category': '可回收物', 'color': 'blue', 'tip': '请投入蓝色可回收垃圾桶'},
    15: {'category': '可回收物', 'color': 'blue', 'tip': '请清洗后投入蓝色可回收垃圾桶'},
    16: {'category': '可回收物', 'color': 'blue', 'tip': '请压扁后投入蓝色可回收垃圾桶'},
    17: {'category': '可回收物', 'color': 'blue', 'tip': '请投入蓝色可回收垃圾桶'},
    18: {'category': '其他垃圾', 'color': 'gray', 'tip': '已使用的餐巾纸请投入灰色其他垃圾桶'},
    19: {'category': '可回收物', 'color': 'blue', 'tip': '请清洗后投入蓝色可回收垃圾桶'},
    20: {'category': '可回收物', 'color': 'blue', 'tip': '请清洗后投入蓝色可回收垃圾桶'},
    21: {'category': '可回收物', 'color': 'blue', 'tip': '请投入蓝色可回收垃圾桶'},
    22: {'category': '可回收物', 'color': 'blue', 'tip': '请清洗后投入蓝色可回收垃圾桶'},
    23: {'category': '可回收物', 'color': 'blue', 'tip': '请清洗后投入蓝色可回收垃圾桶'},
    # 有害垃圾（红色）- ID 24-27
    24: {'category': '有害垃圾', 'color': 'red', 'tip': '请投入红色有害垃圾桶'},
    25: {'category': '有害垃圾', 'color': 'red', 'tip': '请投入红色有害垃圾桶'},
    26: {'category': '有害垃圾', 'color': 'red', 'tip': '请投入红色有害垃圾桶'},
    27: {'category': '有害垃圾', 'color': 'red', 'tip': '请投入红色有害垃圾桶'},
    # 其他垃圾（灰色）- ID 28
    28: {'category': '其他垃圾', 'color': 'gray', 'tip': '请投入灰色其他垃圾桶'},
}
