import sys
import cv2
import os
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QLineEdit, QMessageBox, QLabel
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5 import QtCore
from utils.camera import cameraThread
from utils.face import detectionThread, compareThread
import pickle
import hashlib


class uiWindow(QWidget):
    # signal
    found_face_signal = QtCore.pyqtSignal()

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

        self.line1 = QLineEdit(self)
        self.line1.setGeometry(100+80+10, 10+720+15, 160, 20)
        self.line1.setText('Msgs')
        self.line1.setReadOnly(True)

        self.line2 = QLineEdit(self)
        self.line2.setGeometry(190+160+10+80, 10+720+15, 80, 20)
        self.line2.setText('Pairs')
        self.line2.setReadOnly(True)

        self.line3 = QLineEdit(self)
        self.line3.setGeometry(360+160+10, 10+720+15, 160, 20)
        self.line3.setText('')

        self.label1 = QLabel(self)
        self.label1.setGeometry(190+160+10, 10+720+15, 60, 20)
        self.label1.setText('User')

        # connection
        self.button1.clicked.connect(self.saveFace)
        self.button2.clicked.connect(self.quit)
        self.found_face_signal.connect(self.compareFace)

        # thread
        self.camera_thread = cameraThread(self)
        self.detection_thread = detectionThread(self)
        self.compare_thread = compareThread(self)

        # current faces
        self.current_faces = {'names': None, 'locs': None, 'encodings': None}

        self.show()
        self.setStreamer(start=True)
    
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

    def saveFace(self):
        if self.line3.text() == '':
            QMessageBox.warning(self, 'Warning', '请输入姓名', QMessageBox.Ok)
            return 
            
        if len(self.camera_thread.dets) > 0:
            encodings = self.camera_thread.dets['encodings']
            name = self.line3.text()
            savedict = {'name': name, 'encodings': encodings}
            if not os.path.exists('./data'):
                os.makedirs('./data')
            
            s = hashlib.sha256()
            s.update(name.encode())
            hash_name = s.hexdigest()

            with open(f'./data/{hash_name}.pkl', 'wb') as fo:
                pickle.dump(savedict, fo)

    def compareFace(self):
        if not self.compare_thread.isRunning():
            self.compare_thread.start()

    def quit(self):
        self.setStreamer(False)
        QtCore.QCoreApplication.instance().quit()
