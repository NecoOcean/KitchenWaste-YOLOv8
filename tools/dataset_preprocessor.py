# dataset_preprocessor.py
# -*- coding: utf-8 -*-
"""
数据集完整预处理工具
解决问题：
1. 图片-标签不匹配（清理孤立标签）
2. 验证集图片缺失（重新划分）
3. 类别ID不一致（重映射或扩展配置）
4. 类别不均衡（统计分析）
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
import argparse

# ============== 配置区 ==============

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据集根目录
DATASET_ROOT = PROJECT_ROOT / 'resources' / 'datasets'

# 源数据路径
SRC_IMAGES_TRAIN = DATASET_ROOT / 'images' / 'train'
SRC_IMAGES_VAL = DATASET_ROOT / 'images' / 'val'
SRC_LABELS_TRAIN = DATASET_ROOT / 'labels' / 'train'
SRC_LABELS_VAL = DATASET_ROOT / 'labels' / 'val'

# 输出路径
OUTPUT_DIR = DATASET_ROOT / 'processed'

# 验证集比例
VAL_RATIO = 0.2

# 随机种子
RANDOM_SEED = 42

# 实际使用的40个类别（根据标签分析结果）
FULL_CLASS_NAMES = {
    0: 'vegetable_leaves',      # 菜叶
    1: 'vegetable_roots',       # 菜根
    2: 'fruit_peel',            # 果皮
    3: 'fruit_core',            # 果核
    4: 'bone',                  # 骨头
    5: 'meat_skin',             # 肉皮
    6: 'offal',                 # 内脏
    7: 'rice',                  # 米饭
    8: 'noodles',               # 面条
    9: 'bread_crumbs',          # 面包屑
    10: 'tea_leaves',           # 茶叶渣
    11: 'coffee_grounds',       # 咖啡渣
    12: 'eggshell',             # 蛋壳
    13: 'plastic_bag',          # 食品袋
    14: 'plastic_wrap',         # 保鲜膜
    15: 'plastic_container',    # 塑料盒
    16: 'paper_box',            # 纸盒
    17: 'wrapping_paper',       # 包装纸
    18: 'tissue',               # 餐巾纸
    19: 'zip_top_can',          # 易拉罐
    20: 'tin_can',              # 罐头盒
    21: 'aluminum_foil',        # 铝箔
    22: 'seasoning_bottle',     # 调料瓶
    23: 'wine_bottle',          # 酒瓶
    24: 'expired_seasoning',    # 过期调料
    25: 'expired_medicine',     # 过期药品
    26: 'cleaner_container',    # 清洁剂容器
    27: 'battery',              # 电池
    28: 'other_garbage',        # 其他垃圾
    29: 'class_29',             # 扩展类别
    30: 'class_30',
    31: 'class_31',
    32: 'class_32',
    33: 'class_33',
    34: 'class_34',
    35: 'class_35',
    36: 'class_36',
    37: 'class_37',
    38: 'class_38',
    39: 'class_39',
}

# 类别到垃圾分类的映射
CLASS_CATEGORY = {
    0: '厨余垃圾', 1: '厨余垃圾', 2: '厨余垃圾', 3: '厨余垃圾',
    4: '厨余垃圾', 5: '厨余垃圾', 6: '厨余垃圾', 7: '厨余垃圾',
    8: '厨余垃圾', 9: '厨余垃圾', 10: '厨余垃圾', 11: '厨余垃圾', 12: '厨余垃圾',
    13: '可回收物', 14: '可回收物', 15: '可回收物', 16: '可回收物',
    17: '可回收物', 18: '可回收物', 19: '可回收物', 20: '可回收物',
    21: '可回收物', 22: '可回收物', 23: '可回收物',
    24: '有害垃圾', 25: '有害垃圾', 26: '有害垃圾', 27: '有害垃圾',
    28: '其他垃圾',
    29: '其他垃圾', 30: '其他垃圾', 31: '其他垃圾', 32: '其他垃圾',
    33: '其他垃圾', 34: '其他垃圾', 35: '其他垃圾', 36: '其他垃圾',
    37: '其他垃圾', 38: '其他垃圾', 39: '其他垃圾',
}

# ============== 工具函数 ==============

def get_image_extensions():
    return ['.jpg', '.jpeg', '.png', '.bmp', '.webp']


def find_matching_image(label_path, images_dir):
    """查找标签对应的图片文件"""
    stem = label_path.stem
    for ext in get_image_extensions():
        img_path = images_dir / (stem + ext)
        if img_path.exists():
            return img_path
    return None


def analyze_dataset():
    """分析当前数据集状态"""
    print("\n" + "=" * 60)
    print("步骤 1: 数据集分析")
    print("=" * 60)
    
    stats = {
        'train_images': 0,
        'train_labels': 0,
        'val_images': 0,
        'val_labels': 0,
        'train_matched': 0,
        'val_matched': 0,
        'train_orphan_labels': [],
        'val_orphan_labels': [],
        'class_counts': defaultdict(int),
        'all_classes': set(),
    }
    
    # 统计训练集
    if SRC_IMAGES_TRAIN.exists():
        stats['train_images'] = len(list(SRC_IMAGES_TRAIN.glob('*.*')))
    if SRC_LABELS_TRAIN.exists():
        for label_file in SRC_LABELS_TRAIN.glob('*.txt'):
            stats['train_labels'] += 1
            if find_matching_image(label_file, SRC_IMAGES_TRAIN):
                stats['train_matched'] += 1
                # 统计类别
                with open(label_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            stats['class_counts'][class_id] += 1
                            stats['all_classes'].add(class_id)
            else:
                stats['train_orphan_labels'].append(label_file.name)
    
    # 统计验证集
    if SRC_IMAGES_VAL.exists():
        stats['val_images'] = len(list(SRC_IMAGES_VAL.glob('*.*')))
    if SRC_LABELS_VAL.exists():
        for label_file in SRC_LABELS_VAL.glob('*.txt'):
            stats['val_labels'] += 1
            if find_matching_image(label_file, SRC_IMAGES_VAL):
                stats['val_matched'] += 1
            else:
                stats['val_orphan_labels'].append(label_file.name)
    
    # 输出分析结果
    print(f"\n📊 训练集:")
    print(f"   图片: {stats['train_images']}")
    print(f"   标签: {stats['train_labels']}")
    print(f"   匹配: {stats['train_matched']}")
    print(f"   孤立标签: {len(stats['train_orphan_labels'])}")
    
    print(f"\n📊 验证集:")
    print(f"   图片: {stats['val_images']}")
    print(f"   标签: {stats['val_labels']}")
    print(f"   匹配: {stats['val_matched']}")
    print(f"   孤立标签: {len(stats['val_orphan_labels'])}")
    
    print(f"\n📊 类别统计:")
    print(f"   实际使用类别数: {len(stats['all_classes'])}")
    print(f"   类别ID范围: {min(stats['all_classes'])} - {max(stats['all_classes'])}")
    
    print(f"\n📊 类别分布 (前10):")
    sorted_classes = sorted(stats['class_counts'].items(), key=lambda x: x[1], reverse=True)[:10]
    for class_id, count in sorted_classes:
        name = FULL_CLASS_NAMES.get(class_id, f'unknown_{class_id}')
        print(f"   {class_id:2d}: {count:>5} ({name})")
    
    return stats


def clean_orphan_labels(stats, dry_run=True):
    """清理孤立的标签文件"""
    print("\n" + "=" * 60)
    print("步骤 2: 清理孤立标签" + (" [模拟运行]" if dry_run else ""))
    print("=" * 60)
    
    total_removed = 0
    
    # 清理训练集孤立标签
    if stats['train_orphan_labels']:
        print(f"\n训练集孤立标签: {len(stats['train_orphan_labels'])}")
        for label_name in stats['train_orphan_labels']:
            label_path = SRC_LABELS_TRAIN / label_name
            if not dry_run and label_path.exists():
                label_path.unlink()
            total_removed += 1
    
    # 清理验证集孤立标签
    if stats['val_orphan_labels']:
        print(f"验证集孤立标签: {len(stats['val_orphan_labels'])}")
        for label_name in stats['val_orphan_labels']:
            label_path = SRC_LABELS_VAL / label_name
            if not dry_run and label_path.exists():
                label_path.unlink()
            total_removed += 1
    
    print(f"\n{'将删除' if dry_run else '已删除'} {total_removed} 个孤立标签文件")
    return total_removed


def redistribute_dataset(dry_run=True):
    """重新划分训练集和验证集"""
    print("\n" + "=" * 60)
    print("步骤 3: 重新划分数据集" + (" [模拟运行]" if dry_run else ""))
    print("=" * 60)
    
    # 收集所有有效的图片-标签对
    valid_pairs = []
    
    # 从训练集收集
    if SRC_LABELS_TRAIN.exists():
        for label_file in SRC_LABELS_TRAIN.glob('*.txt'):
            img_file = find_matching_image(label_file, SRC_IMAGES_TRAIN)
            if img_file:
                valid_pairs.append((img_file, label_file, 'train'))
    
    # 从验证集收集
    if SRC_LABELS_VAL.exists():
        for label_file in SRC_LABELS_VAL.glob('*.txt'):
            img_file = find_matching_image(label_file, SRC_IMAGES_VAL)
            if img_file:
                valid_pairs.append((img_file, label_file, 'val'))
    
    print(f"\n有效图片-标签对总数: {len(valid_pairs)}")
    
    # 随机打乱并划分
    random.seed(RANDOM_SEED)
    random.shuffle(valid_pairs)
    
    val_count = int(len(valid_pairs) * VAL_RATIO)
    train_count = len(valid_pairs) - val_count
    
    train_pairs = valid_pairs[:train_count]
    val_pairs = valid_pairs[train_count:]
    
    print(f"新训练集: {len(train_pairs)} ({(1-VAL_RATIO)*100:.0f}%)")
    print(f"新验证集: {len(val_pairs)} ({VAL_RATIO*100:.0f}%)")
    
    if not dry_run:
        # 创建输出目录
        out_train_images = OUTPUT_DIR / 'images' / 'train'
        out_train_labels = OUTPUT_DIR / 'labels' / 'train'
        out_val_images = OUTPUT_DIR / 'images' / 'val'
        out_val_labels = OUTPUT_DIR / 'labels' / 'val'
        
        for d in [out_train_images, out_train_labels, out_val_images, out_val_labels]:
            d.mkdir(parents=True, exist_ok=True)
        
        # 复制训练集
        print("\n复制训练集...")
        for img, lbl, _ in train_pairs:
            shutil.copy2(img, out_train_images / img.name)
            shutil.copy2(lbl, out_train_labels / lbl.name)
        
        # 复制验证集
        print("复制验证集...")
        for img, lbl, _ in val_pairs:
            shutil.copy2(img, out_val_images / img.name)
            shutil.copy2(lbl, out_val_labels / lbl.name)
        
        print(f"\n✅ 数据已输出到: {OUTPUT_DIR}")
    
    return train_count, val_count


def generate_data_yaml(num_classes=40):
    """生成data.yaml配置文件"""
    print("\n" + "=" * 60)
    print("步骤 4: 生成配置文件")
    print("=" * 60)
    
    yaml_path = OUTPUT_DIR / 'data.yaml'
    
    yaml_content = f'''# 厨房垃圾分类数据集配置
# 自动生成 - 包含{num_classes}个类别

path: {OUTPUT_DIR.as_posix()}
train: images/train
val: images/val

nc: {num_classes}

names:
'''
    for i in range(num_classes):
        name = FULL_CLASS_NAMES.get(i, f'class_{i}')
        yaml_content += f'  {i}: {name}\n'
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"✅ 已生成: {yaml_path}")
    return yaml_path


def generate_config_py(num_classes=40):
    """生成Config.py配置文件"""
    config_path = OUTPUT_DIR / 'Config.py'
    
    # 构建类别名称字典
    names = {i: FULL_CLASS_NAMES.get(i, f'class_{i}') for i in range(num_classes)}
    ch_names = list(names.values())
    
    # 构建分类指导
    classification_guide = {}
    color_map = {'厨余垃圾': 'green', '可回收物': 'blue', '有害垃圾': 'red', '其他垃圾': 'gray'}
    tip_map = {
        '厨余垃圾': '请投入绿色厨余垃圾桶',
        '可回收物': '请清洗后投入蓝色可回收垃圾桶',
        '有害垃圾': '请投入红色有害垃圾桶',
        '其他垃圾': '请投入灰色其他垃圾桶'
    }
    for i in range(num_classes):
        category = CLASS_CATEGORY.get(i, '其他垃圾')
        classification_guide[i] = {
            'category': category,
            'color': color_map[category],
            'tip': tip_map[category]
        }
    
    config_content = f'''# Config.py
# -*- coding: utf-8 -*-
"""
基于YOLOv8的垃圾目标检测算法 - 配置文件
自动生成 - {num_classes}类别
"""

# 图片及视频检测结果保存路径
save_path = 'save_data'

# 使用的模型路径
model_path = 'models/best.pt'

# 类别数量
NUM_CLASSES = {num_classes}

# 类别配置
names = {repr(names)}

# 中文类别名称
CH_names = {repr(ch_names)}

# 垃圾分类指导映射
classification_guide = {repr(classification_guide)}
'''
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"✅ 已生成: {config_path}")
    return config_path


def main():
    parser = argparse.ArgumentParser(description='数据集预处理工具')
    parser.add_argument('--execute', action='store_true', help='执行实际操作（默认为模拟运行）')
    parser.add_argument('--output', type=str, default='datasets/processed', help='输出目录')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='验证集比例')
    args = parser.parse_args()
    
    global OUTPUT_DIR, VAL_RATIO
    OUTPUT_DIR = Path(args.output)
    VAL_RATIO = args.val_ratio
    
    dry_run = not args.execute
    
    print("=" * 60)
    print("🔧 数据集预处理工具")
    print("=" * 60)
    print(f"模式: {'实际执行' if args.execute else '模拟运行 (添加 --execute 执行实际操作)'}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"验证集比例: {VAL_RATIO*100:.0f}%")
    
    # 1. 分析数据集
    stats = analyze_dataset()
    
    # 2. 清理孤立标签
    if stats['train_orphan_labels'] or stats['val_orphan_labels']:
        clean_orphan_labels(stats, dry_run=dry_run)
    
    # 3. 重新划分数据集
    train_count, val_count = redistribute_dataset(dry_run=dry_run)
    
    # 4. 生成配置文件
    if not dry_run:
        num_classes = max(stats['all_classes']) + 1 if stats['all_classes'] else 40
        generate_data_yaml(num_classes)
        generate_config_py(num_classes)
    
    # 输出总结
    print("\n" + "=" * 60)
    print("📋 预处理总结")
    print("=" * 60)
    print(f"原始训练集: {stats['train_matched']} 有效对")
    print(f"原始验证集: {stats['val_matched']} 有效对")
    print(f"处理后训练集: {train_count}")
    print(f"处理后验证集: {val_count}")
    print(f"孤立标签: {len(stats['train_orphan_labels']) + len(stats['val_orphan_labels'])} 个")
    
    if dry_run:
        print("\n⚠️  这是模拟运行，未执行实际操作")
        print("运行以下命令执行实际预处理:")
        print(f"  python dataset_preprocessor.py --execute --output {OUTPUT_DIR}")
    else:
        print("\n✅ 预处理完成!")
        print("\n下一步操作:")
        print(f"1. 检查输出目录: {OUTPUT_DIR}")
        print(f"2. 复制配置文件: copy {OUTPUT_DIR}\\Config.py .\\Config.py")
        print(f"3. 修改train.py使用新数据集: data='{OUTPUT_DIR}/data.yaml'")
        print("4. 运行训练: python train.py")


if __name__ == '__main__':
    main()
