#優化性能
# git remote add origin https://github.com/cakeiiii/your_repo.git

import os
import cv2
import pandas as pd
from ultralytics import YOLO, solutions
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox,
    QFileDialog, QComboBox, QProgressBar, QSplitter, QFrame, QSlider
)
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot, Qt, QMutex, QMutexLocker, QTimer
from PyQt6.QtGui import QPixmap, QImage, QPalette, QColor
from collections import defaultdict


class VideoProcessingThread(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(str)
    frame_signal = pyqtSignal(QImage)

    def __init__(self, folder_path, output_path, model_name, mode, line_height_ratio, line_orientation, frame_interval=15):
        super().__init__()
        self.folder_path = folder_path
        self.output_path = output_path
        self.model_name = model_name
        self.mode = mode
        self.line_height_ratio = float(line_height_ratio)
        self.line_orientation = line_orientation
        self.model = YOLO(self.model_name)
        self.frame_height = None
        self.frame_width = None
        self.frame_interval = frame_interval
        self.is_paused = False
        self.is_canceled = False
        self.mutex = QMutex()

    def run(self):
        try:
            video_files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
            total_videos = len(video_files)

            for index, video_file in enumerate(video_files):
                if self.is_canceled:
                    self.finished_signal.emit("处理已取消")
                    return
                if not self.process_video(os.path.join(self.folder_path, video_file)):
                    self.finished_signal.emit("视频处理失败")
                    return
                self.progress_signal.emit(int((index + 1) / total_videos * 100))

            self.finished_signal.emit("所有视频处理完成")
        except Exception as e:
            self.finished_signal.emit(f"处理过程中发生错误: {str(e)}")

    def create_processor(self):
        line_height = self.line_height_ratio * self.frame_height
        line_points = [(1, line_height), (9999, line_height)] if self.line_orientation == '横向' else [(line_height, 1), (line_height, 9999)]
        if self.mode == "物体测速模式":
            return solutions.SpeedEstimator(view_img=False, reg_pts=line_points, names=self.model.names)
        elif self.mode == "物体分布模式":
            return solutions.Heatmap(colormap=cv2.COLORMAP_PARULA, view_img=False, shape="circle", names=self.model.names, decay_factor=1)
        elif self.mode == "物体计数模式":
            return solutions.ObjectCounter(view_img=False, reg_pts=line_points, names=self.model.names, draw_tracks=True, line_thickness=2)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def initialize_video_processing(self, video_file):
        video_path = os.path.join(self.folder_path, video_file)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Error reading video file")
        w, h, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FPS))
        self.frame_height = h
        self.frame_width = w
        output_filename = os.path.splitext(os.path.basename(video_file))[0] + ".avi"
        output_video_path = os.path.join(self.output_path, output_filename)
        video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        return cap, video_writer, output_video_path

    def process_video(self, video_file):
        try:
            cap, video_writer, output_video_path = self.initialize_video_processing(video_file)
            frame_count = 0
            self.processor = self.create_processor()
            detection_results = []
            all_keys = set()

            while cap.isOpened():
                if self.is_canceled:
                    break

                with QMutexLocker(self.mutex):
                    if self.is_paused:
                        self.msleep(100)
                        continue

                success, im0 = cap.read()
                if not success:
                    break

                frame_count += 1
                if frame_count % self.frame_interval == 0:
                    tracks = self.model.track(im0, persist=True, show=False)
                    im0, frame_data = self.process_frame(im0, tracks, frame_count)
                    detection_results.append(frame_data)
                    all_keys.update(frame_data.keys())

                    if frame_count % (self.frame_interval * 5) == 0:  # Reduce the frequency of UI updates
                        qt_image = self.convert_to_qt_format(im0)
                        self.frame_signal.emit(qt_image)

                video_writer.write(im0)

            cap.release()
            video_writer.release()

            self.save_results(detection_results, all_keys, output_video_path)
            return True
        except Exception as e:
            print(f"处理视频 {video_file} 时发生错误: {str(e)}")
            return False

    def convert_to_qt_format(self, im0):
        # 转换图像以适应显示区域
        im0_rgb = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
        qt_image = QImage(im0_rgb.data, im0_rgb.shape[1], im0_rgb.shape[0], QImage.Format.Format_RGB888)
        scaled_image = qt_image.scaled(480, 360, Qt.AspectRatioMode.KeepAspectRatio)
        return scaled_image

    def process_frame(self, im0, tracks, frame_count):
        # 对每一帧进行处理并返回处理后的图像和数据
        if isinstance(self.processor, solutions.Heatmap):
            im0 = self.processor.generate_heatmap(im0, tracks)
            object_count = defaultdict(int)
            for track in tracks:
                for obj in track.boxes:
                    object_count[self.model.names[obj.cls.item()]] += 1
            return im0, {"frame": frame_count, **object_count}

        elif isinstance(self.processor, solutions.SpeedEstimator):
            im0 = self.processor.estimate_speed(im0, tracks)
            frame_speed_data = self.processor.get_frame_speed_data() if self.processor.dist_data else {}
            return im0, {"frame": frame_count, **frame_speed_data}

        elif isinstance(self.processor, solutions.ObjectCounter):
            im0 = self.processor.start_counting(im0, tracks)
            current_counts = self.processor.get_counts()
            return im0, {"frame": frame_count, **current_counts}

    def save_results(self, detection_results, all_keys, output_video_path):
        all_keys.discard('frame')
        columns_order = ['frame'] + list(sorted(all_keys))
        df = pd.DataFrame(detection_results)
        if not df.empty:
            df = df.reindex(columns=columns_order)
        output_csv_path = os.path.splitext(output_video_path)[0] + ".csv"
        df.to_csv(output_csv_path, index=False)

    def pause(self):
        with QMutexLocker(self.mutex):
            self.is_paused = not self.is_paused

    def cancel(self):
        with QMutexLocker(self.mutex):
            self.is_canceled = True


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.updateFrameThrottled)
        self.thread = None
        self.resize(900, 600)
        self.setFixedSize(self.size())

    def initUI(self):
        self.setWindowTitle('视频处理工具')

        # 设置整体颜色主题
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(240, 240, 240))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(0, 0, 0))
        self.setPalette(palette)

        main_layout = QHBoxLayout()

        # 左侧控制面板布局
        control_layout = QVBoxLayout()
        control_layout.setSpacing(5)
        control_layout.setContentsMargins(5, 5, 5, 5)

        # 选择输入文件
        control_layout.addWidget(self.createLabel('选择输入文件:'))
        input_button = self.createButton('选择输入文件夹', self.selectFolder)
        self.folderPath = self.createScrollableLabel('#####文件路径#####')
        control_layout.addWidget(input_button)
        control_layout.addWidget(self.folderPath)

        # 选择输出文件
        control_layout.addWidget(self.createLabel('选择输出文件:'))
        output_button = self.createButton('选择输出文件夹', self.selectOutputFolder)
        self.outputPath = self.createScrollableLabel('#####文件路径#####')
        control_layout.addWidget(output_button)
        control_layout.addWidget(self.outputPath)

        # 选择模式
        control_layout.addWidget(self.createLabel('选择模式:'))
        self.mode_buttons = {}
        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(1)
        mode_layout.addWidget(self.createModeButton('物体分布模式'))
        mode_layout.addWidget(self.createModeButton('物体计数模式'))
        mode_layout.addWidget(self.createModeButton('物体测速模式'))
        control_layout.addLayout(mode_layout)
        self.selected_mode = self.createLabel('当前模式：未选择')
        control_layout.addWidget(self.selected_mode)

        # 分割线
        control_layout.addWidget(self.createSeparator())

        # 调整检测线
        control_layout.addWidget(self.createLabel('调整检测线:'))
        orientation_layout = QHBoxLayout()
        orientation_layout.setSpacing(2)
        self.horizontalButton = self.createButton('横向', lambda: self.selectOrientation('横向'))
        self.verticalButton = self.createButton('纵向', lambda: self.selectOrientation('纵向'))
        orientation_layout.addWidget(self.horizontalButton)
        orientation_layout.addWidget(self.verticalButton)
        control_layout.addLayout(orientation_layout)
        self.selected_orientation = self.createLabel('当前检测线方向：未选择')
        control_layout.addWidget(self.selected_orientation)

        self.linePositionSlider = QSlider(Qt.Orientation.Horizontal)
        control_layout.addWidget(self.linePositionSlider)

        # 分割线
        control_layout.addWidget(self.createSeparator())

        # 选择模型
        model_layout = QHBoxLayout()
        model_layout.setSpacing(2)
        self.modelNameEdit = QLineEdit('yolov8n.pt')
        model_layout.addWidget(self.createLabel('选择模型  :'))
        model_layout.addWidget(self.modelNameEdit)
        control_layout.addLayout(model_layout)

        # 自定义模型输入
        custom_model_layout = QHBoxLayout()
        custom_model_layout.setSpacing(2)
        custom_model_layout.addWidget(self.createLabel('自定义模型:'))
        self.customModelEdit = QLineEdit()
        custom_model_layout.addWidget(self.customModelEdit)
        control_layout.addLayout(custom_model_layout)

        # 选择检测精度
        frame_interval_layout = QHBoxLayout()
        frame_interval_layout.setSpacing(2)
        self.frameIntervalLabel = self.createLabel(f'帧间隔: {1}')
        self.frameIntervalSlider = QSlider(Qt.Orientation.Horizontal)
        self.frameIntervalSlider.setRange(1, 20)
        self.frameIntervalSlider.setValue(1)
        self.frameIntervalSlider.valueChanged.connect(self.updateFrameIntervalLabel)
        frame_interval_layout.addWidget(self.frameIntervalLabel)
        frame_interval_layout.addWidget(self.frameIntervalSlider)
        control_layout.addLayout(frame_interval_layout)

        control_frame = QFrame()
        control_frame.setLayout(control_layout)

        # 右侧视频显示区域
        video_layout = QVBoxLayout()
        video_layout.setSpacing(2)
        video_layout.setContentsMargins(5, 5, 5, 5)
        self.video_label = QLabel("视频渲染区域")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setFixedSize(640, 360)
        video_layout.addWidget(self.video_label)

        # 底部控制按钮和进度条
        button_layout = QHBoxLayout()
        button_layout.setSpacing(2)
        button_layout.addWidget(self.createButton('开始', self.startProcessing))
        button_layout.addWidget(self.createButton('暂停/继续', self.pauseProcessing))
        button_layout.addWidget(self.createButton('取消', self.cancelProcessing))
        video_layout.addLayout(button_layout)

        self.progressBar = QProgressBar()
        video_layout.addWidget(self.progressBar)

        video_frame = QFrame()
        video_frame.setLayout(video_layout)

        # 使用分割器进行左右布局
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(control_frame)
        splitter.addWidget(video_frame)
        splitter.setStretchFactor(1, 3)

        main_layout.addWidget(splitter)

        self.setLayout(main_layout)
        self.pending_frame = None

    def createButton(self, text, callback):
        button = QPushButton(text)
        button.clicked.connect(callback)
        button.setStyleSheet("""
            QPushButton {
                background-color: #E0E0E0; 
                border: 1px solid #CCCCCC; 
                padding: 3px 8px;
                margin: 0px;
            }
            QPushButton:hover {
                background-color: #B0C4DE;
            }
        """)
        return button

    def createModeButton(self, mode):
        button = self.createButton(mode, lambda: self.selectMode(mode))
        self.mode_buttons[mode] = button
        return button

    def createLabel(self, text):
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        label.setStyleSheet("color: black; margin: 0px;")
        return label

    def createScrollableLabel(self, text):
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        label.setWordWrap(True)
        label.setStyleSheet("color: black; margin: 0px;")
        return label

    def createSeparator(self):
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("margin-top: 5px; margin-bottom: 5px;")
        return line

    def updateFrameIntervalLabel(self, value):
        self.frameIntervalLabel.setText(f'帧间隔: {value}')

    def selectFolder(self):
        folder = QFileDialog.getExistingDirectory(self, '选择文件夹')
        if folder:
            self.folderPath.setText(folder)

    def selectOutputFolder(self):
        folder = QFileDialog.getExistingDirectory(self, '选择文件夹')
        if folder:
            self.outputPath.setText(folder)

    def selectMode(self, mode):
        self.selected_mode.setText(f'当前模式：{mode}')
        for btn_mode, btn in self.mode_buttons.items():
            if btn_mode == mode:
                btn.setStyleSheet("background-color: #B0C4DE; border: 1px solid #CCCCCC; padding: 3px 8px;")
            else:
                btn.setStyleSheet("background-color: #E0E0E0; border: 1px solid #CCCCCC; padding: 3px 8px;")

    def selectOrientation(self, orientation):
        self.selected_orientation.setText(f'当前检测线方向：{orientation}')
        if orientation == '横向':
            self.horizontalButton.setStyleSheet("background-color: #B0C4DE; border: 1px solid #CCCCCC; padding: 3px 8px;")
            self.verticalButton.setStyleSheet("background-color: #E0E0E0; border: 1px solid #CCCCCC; padding: 3px 8px;")
        else:
            self.verticalButton.setStyleSheet("background-color: #B0C4DE; border: 1px solid #CCCCCC; padding: 3px 8px;")
            self.horizontalButton.setStyleSheet("background-color: #E0E0E0; border: 1px solid #CCCCCC; padding: 3px 8px;")

    def startProcessing(self):
        if self.thread is not None and self.thread.isRunning():
            QMessageBox.warning(self, '警告', '当前处理尚未完成，请等待。')
            return

        folder_path = self.folderPath.text()
        output_path = self.outputPath.text()
        model_name = self.customModelEdit.text() or self.modelNameEdit.text()
        mode = self.selected_mode.text().replace('当前模式：', '')
        line_orientation = '横向' if '横向' in self.selected_orientation.text() else '纵向'
        frame_interval = self.frameIntervalSlider.value()

        self.thread = VideoProcessingThread(folder_path, output_path, model_name, mode, self.linePositionSlider.value() / 100, line_orientation, frame_interval)
        self.thread.progress_signal.connect(self.progressBar.setValue)
        self.thread.finished_signal.connect(self.finishedProcessing)
        self.thread.frame_signal.connect(self.enqueueFrame)
        self.thread.start()
        self.timer.start()

    @pyqtSlot(QImage)
    def enqueueFrame(self, image):
        self.pending_frame = image

    @pyqtSlot()
    def updateFrameThrottled(self):
        if self.pending_frame:
            self.video_label.setPixmap(QPixmap.fromImage(self.pending_frame))
            self.pending_frame = None

    @pyqtSlot()
    def pauseProcessing(self):
        if hasattr(self, 'thread'):
            self.thread.pause()

    @pyqtSlot()
    def cancelProcessing(self):
        if hasattr(self, 'thread'):
            self.thread.cancel()

    def finishedProcessing(self, message):
        self.timer.stop()
        QMessageBox.information(self, '处理结果', message)
        self.thread = None


if __name__ == '__main__':
    app = QApplication([])
    mainWindow = MainWindow()
    mainWindow.show()
    app.exec()
