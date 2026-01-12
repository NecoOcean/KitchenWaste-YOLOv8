# ui_manager.py
# -*- coding: utf-8 -*-
"""
基于YOLOv8的垃圾目标检测算法 - UI管理模块
"""
import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QTableWidgetItem
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


class UIManager:
    """UI界面管理类"""
    
    def __init__(self, ui):
        self.ui = ui
    
    def display_image(self, image: np.ndarray, label, keep_ratio: bool = True):
        """在QLabel上显示图像"""
        if image is None:
            return
        
        # BGR转RGB
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        h, w, ch = image_rgb.shape
        bytes_per_line = ch * w
        
        # 创建QImage
        q_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # 缩放以适应标签大小
        if keep_ratio:
            scaled_pixmap = pixmap.scaled(
                label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        else:
            scaled_pixmap = pixmap.scaled(label.size(), Qt.IgnoreAspectRatio)
        
        label.setPixmap(scaled_pixmap)
    
    def update_result_table(self, detection_result, table_widget):
        """更新检测结果表格"""
        table_widget.setRowCount(0)
        
        if not detection_result.has_detections:
            return
        
        for i, cls_id in enumerate(detection_result.classes):
            row = table_widget.rowCount()
            table_widget.insertRow(row)
            
            # 类别名称
            ch_name = Config.CH_names[cls_id] if cls_id < len(Config.CH_names) else f'类别{cls_id}'
            table_widget.setItem(row, 0, QTableWidgetItem(ch_name))
            
            # 置信度
            table_widget.setItem(row, 1, QTableWidgetItem(detection_result.confidence_strings[i]))
            
            # 位置坐标
            if i < len(detection_result.locations):
                loc = detection_result.locations[i]
                loc_str = f"({loc[0]}, {loc[1]}, {loc[2]}, {loc[3]})"
                table_widget.setItem(row, 2, QTableWidgetItem(loc_str))
            
            # 分类指导
            guide = Config.classification_guide.get(cls_id, {})
            category = guide.get('category', '未知')
            table_widget.setItem(row, 3, QTableWidgetItem(category))
    
    def update_detection_info(self, detection_result, info_label):
        """更新检测信息标签"""
        if detection_result.has_detections:
            info_text = f"检测到 {detection_result.count} 个目标 | 耗时: {detection_result.elapsed_time*1000:.1f}ms"
        else:
            info_text = f"未检测到目标 | 耗时: {detection_result.elapsed_time*1000:.1f}ms"
        info_label.setText(info_text)
    
    def show_classification_guide(self, guides: list, guide_label):
        """显示分类指导信息"""
        if not guides:
            guide_label.setText("暂无分类指导")
            return
        
        guide_texts = []
        for guide in guides:
            text = f"【{guide['name']}】→ {guide['category']}: {guide['tip']}"
            guide_texts.append(text)
        
        guide_label.setText('\n'.join(guide_texts))
    
    def clear_display(self, label):
        """清空显示"""
        label.clear()
        label.setText("请选择图片、视频或开启摄像头")
    
    def set_status(self, status_label, text: str, is_error: bool = False):
        """设置状态信息"""
        if is_error:
            status_label.setStyleSheet("color: red;")
        else:
            status_label.setStyleSheet("color: green;")
        status_label.setText(text)
    
    def update_combobox(self, combobox, items: list, current_index: int = 0):
        """更新下拉框选项"""
        combobox.clear()
        combobox.addItems(items)
        if 0 <= current_index < len(items):
            combobox.setCurrentIndex(current_index)
    
    def get_color_style(self, color_name: str) -> str:
        """获取颜色样式"""
        color_map = {
            'green': '#4CAF50',
            'blue': '#2196F3',
            'red': '#F44336',
            'gray': '#9E9E9E'
        }
        return color_map.get(color_name, '#9E9E9E')
