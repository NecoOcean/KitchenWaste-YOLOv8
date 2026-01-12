# main.py
# -*- coding: utf-8 -*-
"""
基于YOLOv8的垃圾目标检测算法 - 主程序
"""
import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from UIProgram.UiMain import Ui_MainWindow
from core.detection_service import DetectionService
from core.ui_manager import UIManager
from core.file_handler import FileHandler
from core.statistics_manager import StatisticsManager
from config import Config

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class VideoThread(QThread):
    """视频/摄像头处理线程"""
    frame_signal = pyqtSignal(np.ndarray, object)  # 发送帧和检测结果
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)
    
    def __init__(self, source, detection_service):
        super().__init__()
        self.source = source
        self.detection_service = detection_service
        self.running = True
        self.save_video = False
        self.video_writer = None
        self.save_path = None
    
    def run(self):
        try:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                self.error_signal.emit(f"无法打开视频源: {self.source}")
                return
            
            # 获取视频信息
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 初始化视频写入器
            if self.save_video and self.save_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(self.save_path, fourcc, fps, (width, height))
            
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 执行检测
                result = self.detection_service.detect(frame)
                
                # 获取绘制后的图像
                plotted_frame = result.get_plotted_image()
                
                # 保存视频帧
                if self.video_writer is not None:
                    self.video_writer.write(plotted_frame)
                
                # 发送信号
                self.frame_signal.emit(plotted_frame, result)
                
                # 控制帧率
                self.msleep(int(1000 / fps))
            
            cap.release()
            if self.video_writer is not None:
                self.video_writer.release()
            
        except Exception as e:
            self.error_signal.emit(str(e))
        finally:
            self.finished_signal.emit()
    
    def stop(self):
        self.running = False
    
    def enable_save(self, save_path):
        self.save_video = True
        self.save_path = save_path


class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # 初始化组件
        self.detection_service = None
        self.ui_manager = UIManager(self.ui)
        self.statistics_manager = StatisticsManager()
        self.video_thread = None
        self.current_image = None
        self.current_result = None
        self.image_list = []
        self.current_image_index = 0
        
        # 加载模型
        self._init_model()
        
        # 连接信号槽
        self._connect_signals()
        
        # 初始化状态
        self._init_state()
    
    def _init_model(self):
        """初始化检测模型"""
        try:
            # 检查模型文件是否存在
            if not os.path.exists(Config.model_path):
                # 尝试使用预训练模型
                alt_paths = ['yolov8n.pt', 'models/yolov8n.pt']
                model_found = False
                for path in alt_paths:
                    if os.path.exists(path):
                        Config.model_path = path
                        model_found = True
                        break
                
                if not model_found:
                    QMessageBox.warning(
                        self, "模型警告",
                        f"未找到模型文件: {Config.model_path}\n"
                        "将尝试下载预训练模型yolov8n.pt"
                    )
                    Config.model_path = 'yolov8n.pt'
            
            self.detection_service = DetectionService(Config.model_path)
            self.ui.statusLabel.setText(f"模型加载成功: {Config.model_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型加载失败: {e}")
            self.ui.statusLabel.setText("模型加载失败")
    
    def _connect_signals(self):
        """连接信号槽"""
        self.ui.PicBtn.clicked.connect(self.on_open_image)
        self.ui.FolderBtn.clicked.connect(self.on_open_folder)
        self.ui.VideoBtn.clicked.connect(self.on_open_video)
        self.ui.CapBtn.clicked.connect(self.on_toggle_camera)
        self.ui.SaveBtn.clicked.connect(self.on_save)
        self.ui.StopBtn.clicked.connect(self.on_stop)
        
        # 菜单动作
        self.ui.actionOpen.triggered.connect(self.on_open_image)
        self.ui.actionExit.triggered.connect(self.close)
        self.ui.actionAbout.triggered.connect(self.on_about)
        
        # 统计按钮
        self.ui.exportStatsBtn.clicked.connect(self.on_export_statistics)
        self.ui.clearStatsBtn.clicked.connect(self.on_clear_statistics)
    
    def _init_state(self):
        """初始化状态"""
        self.ui.StopBtn.setEnabled(False)
        self.ui.SaveBtn.setEnabled(False)
        self._update_statistics_display()
    
    def on_open_image(self):
        """打开图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.gif);;所有文件 (*.*)"
        )
        
        if file_path:
            self._detect_image(file_path)
    
    def on_open_folder(self):
        """打开文件夹"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        
        if folder_path:
            self.image_list = FileHandler.get_images_from_directory(folder_path)
            if self.image_list:
                self.current_image_index = 0
                self._detect_image(self.image_list[0])
                self.ui.statusLabel.setText(f"已加载 {len(self.image_list)} 张图片")
            else:
                QMessageBox.information(self, "提示", "文件夹中没有找到图片")
    
    def on_open_video(self):
        """打开视频"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)"
        )
        
        if file_path:
            self._start_video(file_path)
    
    def on_toggle_camera(self):
        """切换摄像头"""
        if self.video_thread and self.video_thread.isRunning():
            self.on_stop()
        else:
            self._start_video(0)  # 默认摄像头
    
    def on_save(self):
        """保存检测结果"""
        if self.current_image is not None:
            save_path = FileHandler.generate_save_path("detected_result.jpg")
            cv2.imwrite(save_path, self.current_image)
            self.ui.statusLabel.setText(f"已保存: {save_path}")
            QMessageBox.information(self, "保存成功", f"结果已保存到:\n{save_path}")
    
    def on_stop(self):
        """停止视频/摄像头"""
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait()
            self.video_thread = None
        
        self.ui.CapBtn.setText("📹 开启摄像头")
        self.ui.StopBtn.setEnabled(False)
        self.ui.statusLabel.setText("已停止")
    
    def on_about(self):
        """关于对话框"""
        QMessageBox.about(
            self, "关于",
            "基于YOLOv8的垃圾目标检测系统\n\n"
            "功能：\n"
            "• 图片垃圾检测\n"
            "• 视频垃圾检测\n"
            "• 实时摄像头检测\n"
            "• 垃圾分类指导\n\n"
            "技术栈：YOLOv8 + PyQt5 + OpenCV"
        )
    
    def _detect_image(self, image_path):
        """检测单张图片"""
        if not self.detection_service:
            QMessageBox.warning(self, "警告", "模型未加载")
            return
        
        try:
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                QMessageBox.warning(self, "警告", f"无法读取图片: {image_path}")
                return
            
            # 执行检测
            result = self.detection_service.detect(image)
            
            # 获取绘制后的图像
            plotted_image = result.get_plotted_image()
            self.current_image = plotted_image
            self.current_result = result
            
            # 更新显示
            self.ui_manager.display_image(plotted_image, self.ui.imageLabel)
            self.ui_manager.update_result_table(result, self.ui.resultTable)
            self.ui_manager.update_detection_info(result, self.ui.detectInfoLabel)
            
            # 显示分类指导
            guides = result.get_classification_guide()
            self.ui_manager.show_classification_guide(guides, self.ui.guideLabel)
            
            # 记录统计
            self.statistics_manager.add_record(result)
            self._update_statistics_display()
            
            # 更新状态
            self.ui.statusLabel.setText(f"检测完成: {os.path.basename(image_path)}")
            self.ui.SaveBtn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"检测失败: {e}")
    
    def _start_video(self, source):
        """启动视频处理"""
        if not self.detection_service:
            QMessageBox.warning(self, "警告", "模型未加载")
            return
        
        # 停止之前的线程
        self.on_stop()
        
        # 创建新线程
        self.video_thread = VideoThread(source, self.detection_service)
        self.video_thread.frame_signal.connect(self._on_video_frame)
        self.video_thread.finished_signal.connect(self._on_video_finished)
        self.video_thread.error_signal.connect(self._on_video_error)
        
        # 启动线程
        self.video_thread.start()
        
        # 更新UI
        if source == 0:
            self.ui.CapBtn.setText("📹 关闭摄像头")
            self.ui.statusLabel.setText("摄像头已开启")
        else:
            self.ui.statusLabel.setText(f"正在播放: {os.path.basename(str(source))}")
        
        self.ui.StopBtn.setEnabled(True)
    
    def _on_video_frame(self, frame, result):
        """处理视频帧"""
        self.current_image = frame
        self.current_result = result
        
        # 更新显示
        self.ui_manager.display_image(frame, self.ui.imageLabel)
        self.ui_manager.update_result_table(result, self.ui.resultTable)
        self.ui_manager.update_detection_info(result, self.ui.detectInfoLabel)
        
        # 显示分类指导
        guides = result.get_classification_guide()
        self.ui_manager.show_classification_guide(guides, self.ui.guideLabel)
        
        # 视频帧统计（每30帧记录一次避免过多记录）
        if not hasattr(self, '_video_frame_count'):
            self._video_frame_count = 0
        self._video_frame_count += 1
        if self._video_frame_count % 30 == 0 and result.has_detections:
            self.statistics_manager.add_record(result)
            self._update_statistics_display()
        
        self.ui.SaveBtn.setEnabled(True)
    
    def _on_video_finished(self):
        """视频处理完成"""
        self.ui.CapBtn.setText("📹 开启摄像头")
        self.ui.StopBtn.setEnabled(False)
        self.ui.statusLabel.setText("视频播放完成")
    
    def _on_video_error(self, error_msg):
        """视频处理错误"""
        QMessageBox.critical(self, "错误", error_msg)
        self.on_stop()
    
    def closeEvent(self, event):
        """关闭事件"""
        self.on_stop()
        event.accept()
    
    def keyPressEvent(self, event):
        """键盘事件"""
        if event.key() == Qt.Key_Escape:
            self.on_stop()
        elif event.key() == Qt.Key_Right and self.image_list:
            # 下一张图片
            self.current_image_index = (self.current_image_index + 1) % len(self.image_list)
            self._detect_image(self.image_list[self.current_image_index])
        elif event.key() == Qt.Key_Left and self.image_list:
            # 上一张图片
            self.current_image_index = (self.current_image_index - 1) % len(self.image_list)
            self._detect_image(self.image_list[self.current_image_index])
    
    def _update_statistics_display(self):
        """更新统计显示"""
        try:
            # 获取今日统计
            today_stats = self.statistics_manager.get_today_statistics()
            self.ui.todayStatsLabel.setText(
                f"今日检测: {today_stats['detection_count']} 次 | 共 {today_stats['total_items']} 项"
            )
            
            # 获取分类统计
            category_stats = self.statistics_manager.get_category_statistics()
            
            self.ui.kitchenWasteLabel.setText(f"🟢 厨余垃圾: {category_stats.get('厨余垃圾', 0)}")
            self.ui.recyclableLabel.setText(f"🔵 可回收物: {category_stats.get('可回收物', 0)}")
            self.ui.hazardousLabel.setText(f"🔴 有害垃圾: {category_stats.get('有害垃圾', 0)}")
            self.ui.otherWasteLabel.setText(f"⚫ 其他垃圾: {category_stats.get('其他垃圾', 0)}")
        except Exception as e:
            print(f"[WARNING] 更新统计显示失败: {e}")
    
    def on_export_statistics(self):
        """导出统计数据"""
        try:
            export_path = self.statistics_manager.export_to_csv()
            if export_path:
                self.ui.statusLabel.setText(f"统计已导出: {export_path}")
                QMessageBox.information(self, "导出成功", f"统计数据已导出到:\n{export_path}")
            else:
                QMessageBox.warning(self, "导出失败", "无法导出统计数据")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败: {e}")
    
    def on_clear_statistics(self):
        """清空统计记录"""
        reply = QMessageBox.question(
            self, "确认清空",
            "确定要清空所有统计记录吗？此操作不可恢复。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.statistics_manager.clear_records()
            self._update_statistics_display()
            self.ui.statusLabel.setText("统计记录已清空")


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle('Fusion')
    
    # 创建主窗口
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
