import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QLineEdit
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5 import QtCore
from utils.camera import cameraThread, detectionThread


class uiWindow(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()
        self.isTerminated = False

    def initUI(self):
        self.setGeometry(50, 50, 1080, 720)
        # self.resize(640, 480)
        self.setWindowTitle('face')

        self.imgLabel = QLabel(self)
        self.imgLabel.setGeometry(10, 10, 620, 460)

        self.button1 = QPushButton(self)
        self.button1.setText('save')
        self.button1.setGeometry(10, 10+720+10, 80, 30)

        self.button2 = QPushButton(self)
        self.button2.setText('quit')
        self.button2.setGeometry(10+80+10, 10+720+10, 80, 30)

        self.line = QLineEdit(self)
        self.line.setGeometry(10+80+10+80+10, 10+720+15, 160, 20)
        self.line.setText('Hello')

        # signal
        self.save_face_signal = QtCore.pyqtSignal()

        # connection
        self.button2.clicked.connect(self.quit)

        # thread
        self.camera_thread = cameraThread(self)
        self.detection_thread = detectionThread(self)

        self.setStreamer(start=True)
        self.show()
    
    def update(self, image):
        h, w, d = image.shape
        self.resize(w + 20, h + 20 + 40)
        self.imgLabel.resize(w, h)
        qimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        qimg = QImage(qimg.data, w, h, w * d, QImage.Format_RGB888)
        self.imgLabel.setPixmap(QPixmap.fromImage(qimg))

    def setStreamer(self, start=False):
        if start:
            self.camera_thread.start()
            self.detection_thread.start()
        else:
            self.camera_thread.close()
            self.close()

    def quit(self):
        self.setStreamer(False)
        QtCore.QCoreApplication.instance().quit()
