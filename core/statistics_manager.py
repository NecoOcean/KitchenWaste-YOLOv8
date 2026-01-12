# statistics_manager.py
# -*- coding: utf-8 -*-
"""
基于YOLOv8的垃圾目标检测算法 - 统计管理模块
"""
import os
import json
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


class StatisticsManager:
    """检测统计管理类"""
    
    def __init__(self, data_file: str = None):
        self.data_file = data_file or os.path.join(Config.save_path, 'statistics.json')
        self.records: List[Dict] = []
        self._ensure_directory()
        self._load_records()
    
    def _ensure_directory(self):
        """确保保存目录存在"""
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
    
    def _load_records(self):
        """加载历史记录"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    self.records = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.records = []
        else:
            self.records = []
    
    def _save_records(self):
        """保存记录到文件"""
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(self.records, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"[ERROR] 保存统计记录失败: {e}")
    
    def add_record(self, detection_result) -> Dict:
        """
        添加检测记录
        :param detection_result: DetectionResult 对象
        :return: 新添加的记录
        """
        if not detection_result.has_detections:
            return None
        
        detected_items = []
        for i, cls_id in enumerate(detection_result.classes):
            ch_name = Config.CH_names[cls_id] if cls_id < len(Config.CH_names) else f'类别{cls_id}'
            guide = Config.classification_guide.get(cls_id, {})
            detected_items.append({
                'class_id': cls_id,
                'name': ch_name,
                'category': guide.get('category', '其他垃圾'),
                'confidence': detection_result.confidences[i] if i < len(detection_result.confidences) else 0
            })
        
        record = {
            'id': len(self.records) + 1,
            'timestamp': datetime.now().isoformat(),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S'),
            'total_count': len(detected_items),
            'items': detected_items,
            'elapsed_time': detection_result.elapsed_time
        }
        
        self.records.append(record)
        self._save_records()
        return record
    
    def get_category_statistics(self) -> Dict[str, int]:
        """
        获取按垃圾分类的统计
        :return: {'厨余垃圾': 10, '可回收物': 5, ...}
        """
        stats = defaultdict(int)
        for record in self.records:
            for item in record.get('items', []):
                category = item.get('category', '其他垃圾')
                stats[category] += 1
        return dict(stats)
    
    def get_class_statistics(self) -> Dict[str, int]:
        """
        获取按具体类别的统计
        :return: {'果皮': 10, '易拉罐': 5, ...}
        """
        stats = defaultdict(int)
        for record in self.records:
            for item in record.get('items', []):
                name = item.get('name', '未知')
                stats[name] += 1
        return dict(stats)
    
    def get_daily_statistics(self) -> Dict[str, int]:
        """
        获取每日检测次数统计
        :return: {'2024-01-01': 10, '2024-01-02': 15, ...}
        """
        stats = defaultdict(int)
        for record in self.records:
            date = record.get('date', 'unknown')
            stats[date] += record.get('total_count', 0)
        return dict(stats)
    
    def get_today_statistics(self) -> Dict[str, Any]:
        """
        获取今日统计摘要
        """
        today = datetime.now().strftime('%Y-%m-%d')
        today_records = [r for r in self.records if r.get('date') == today]
        
        total_items = 0
        category_counts = defaultdict(int)
        
        for record in today_records:
            for item in record.get('items', []):
                total_items += 1
                category_counts[item.get('category', '其他垃圾')] += 1
        
        return {
            'date': today,
            'detection_count': len(today_records),
            'total_items': total_items,
            'category_breakdown': dict(category_counts)
        }
    
    def get_total_statistics(self) -> Dict[str, Any]:
        """
        获取总体统计摘要
        """
        total_items = sum(r.get('total_count', 0) for r in self.records)
        return {
            'total_detections': len(self.records),
            'total_items': total_items,
            'category_statistics': self.get_category_statistics(),
            'class_statistics': self.get_class_statistics()
        }
    
    def get_recent_records(self, limit: int = 10) -> List[Dict]:
        """
        获取最近的检测记录
        """
        return self.records[-limit:][::-1]
    
    def clear_records(self):
        """清空所有记录"""
        self.records = []
        self._save_records()
    
    def export_to_csv(self, export_path: str = None) -> str:
        """
        导出统计数据为CSV
        """
        export_path = export_path or os.path.join(Config.save_path, f'statistics_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        
        try:
            with open(export_path, 'w', encoding='utf-8-sig') as f:
                f.write('记录ID,日期,时间,检测数量,类别详情,垃圾分类,检测耗时(ms)\n')
                for record in self.records:
                    items_str = ';'.join([item['name'] for item in record.get('items', [])])
                    categories_str = ';'.join([item['category'] for item in record.get('items', [])])
                    f.write(f"{record['id']},{record['date']},{record['time']},{record['total_count']},{items_str},{categories_str},{record['elapsed_time']*1000:.1f}\n")
            return export_path
        except IOError as e:
            print(f"[ERROR] 导出CSV失败: {e}")
            return None
