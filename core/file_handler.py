# file_handler.py
# -*- coding: utf-8 -*-
"""
基于YOLOv8的垃圾目标检测算法 - 文件处理模块
"""
import os
from pathlib import Path
from datetime import datetime
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


class FileHandler:
    """文件处理工具类"""
    
    # 支持的图片格式
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}
    
    # 支持的视频格式
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    
    @staticmethod
    def is_image_file(file_path: str) -> bool:
        """判断是否为图片文件"""
        if not file_path:
            return False
        ext = Path(file_path).suffix.lower()
        return ext in FileHandler.IMAGE_EXTENSIONS
    
    @staticmethod
    def is_video_file(file_path: str) -> bool:
        """判断是否为视频文件"""
        if not file_path:
            return False
        ext = Path(file_path).suffix.lower()
        return ext in FileHandler.VIDEO_EXTENSIONS
    
    @staticmethod
    def get_images_from_directory(directory: str) -> list:
        """从目录获取所有图片文件"""
        images = []
        if not os.path.isdir(directory):
            return images
        
        for file_name in os.listdir(directory):
            file_path = os.path.join(directory, file_name)
            if os.path.isfile(file_path) and FileHandler.is_image_file(file_path):
                images.append(file_path)
        
        return sorted(images)
    
    @staticmethod
    def generate_save_path(original_path: str, prefix: str = 'detected') -> str:
        """生成保存路径"""
        # 确保保存目录存在
        save_dir = Config.save_path
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成带时间戳的文件名
        original_name = Path(original_path).stem
        ext = Path(original_path).suffix
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_name = f"{prefix}_{original_name}_{timestamp}{ext}"
        
        return os.path.join(save_dir, new_name)
    
    @staticmethod
    def generate_video_save_path(original_path: str = None) -> str:
        """生成视频保存路径"""
        save_dir = Config.save_path
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if original_path:
            original_name = Path(original_path).stem
            new_name = f"detected_{original_name}_{timestamp}.mp4"
        else:
            new_name = f"camera_record_{timestamp}.mp4"
        
        return os.path.join(save_dir, new_name)
    
    @staticmethod
    def ensure_directory(path: str) -> None:
        """确保目录存在"""
        os.makedirs(path, exist_ok=True)
    
    @staticmethod
    def get_file_info(file_path: str) -> dict:
        """获取文件信息"""
        if not os.path.exists(file_path):
            return {}
        
        stat = os.stat(file_path)
        return {
            'name': Path(file_path).name,
            'size': stat.st_size,
            'size_str': FileHandler._format_size(stat.st_size),
            'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'extension': Path(file_path).suffix.lower()
        }
    
    @staticmethod
    def _format_size(size: int) -> str:
        """格式化文件大小"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} TB"
