import sys
import numpy as np
import cv2  # OpenCV 라이브러리 추가
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('웹캠 이미지 분류기')
        self.setGeometry(100, 100, 400, 300)
        
        layout = QVBoxLayout()
        
        self.imageLabel = QLabel('웹캠에서 이미지를 캡처해주세요.')
        self.imageLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.imageLabel)
        
        self.btnStartWebcam = QPushButton('웹캠 시작')
        self.btnStartWebcam.clicked.connect(self.start_webcam)
        layout.addWidget(self.btnStartWebcam)
        
        self.resultLabel = QLabel('')
        self.resultLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.resultLabel)
        
        self.setLayout(layout)
        
        # 사용자 모델 로드
        self.model = load_model('D:\camkeras\keras_model.h5')
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = None
        
    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(20)
        
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # OpenCV는 이미지를 BGR 형식으로 처리하지만, QPixmap은 RGB 형식을 사용합니다.
            # 따라서 cvtColor 함수를 사용하여 BGR에서 RGB로 변환해야 합니다.
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(200, 200, Qt.KeepAspectRatio)
            self.imageLabel.setPixmap(QPixmap.fromImage(p))
            
            # 이미지 분류
            img_array = cv2.resize(rgb_image, (224, 224))
            img_array_expanded_dims = np.expand_dims(img_array, axis=0)
            preprocessed_image = preprocess_input(img_array_expanded_dims)
            predictions = self.model.predict(preprocessed_image)
            class_names = ['눈', '코', '입']
            predictions_percentages = [round(probability * 100, 2) for probability in predictions[0]]
            result_string = "\n".join([f"{class_names[i]}: {predictions_percentages[i]}%" for i in range(len(class_names))])
            self.resultLabel.setText(f"결과:\n{result_string}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
