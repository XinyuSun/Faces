import sys
import cv2
import os
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QLineEdit, QMessageBox, QLabel, QMainWindow
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5 import QtCore
from utils.camera import cameraThread
from utils.face import detectionThread, compareThread, showThread
from ui.Ui_mainWindow import Ui_MainWindow
import pickle
import hashlib
import time


class uiWindow(Ui_MainWindow, QWidget):
    # signal
    found_face_signal = QtCore.pyqtSignal()
    update_results_signal = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()

        self.icon_file = 'ui/resources/icon1.png'

        self.mainWindow = QMainWindow()
        self.setupUi(self.mainWindow)

        self.roiLabel = [self.roiLabel1, self.roiLabel2, self.roiLabel3, self.roiLabel4]
        self.names = [self.name1, self.name2, self.name3, self.name4]
        for label in self.roiLabel:
            self.setImgLabel(self.icon_file, label)

        # connection
        self.button1.clicked.connect(self.saveFace)
        self.button2.clicked.connect(self.quit)
        self.found_face_signal.connect(self.compareFace)
        self.update_results_signal.connect(self.showFace)

        # thread
        self.camera_thread = cameraThread(self)
        self.detection_thread = detectionThread(self)
        self.compare_thread = compareThread(self)
        self.show_thread = showThread(self)

        # current faces
        self.current_faces = {'names': None, 'locs': None, 'encodings': None, 'scores': None}
        self.cache_faces = []
        self.last_show_time = 0.0

        self.setStreamer(start=True)
        self.mainWindow.show()

    def update(self, image):
        h, w, d = image.shape
        self.mainWindow.resize(w + 20, h + 20 + 40)
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
        if self.line1.text() == '':
            QMessageBox.warning(self, 'Warning', '请输入姓名', QMessageBox.Ok)
            return 
            
        if len(self.camera_thread.dets) > 0:
            encodings = self.camera_thread.dets['encodings']
            name = self.line1.text()
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

    def showFace(self):
        if not self.show_thread.isRunning():
            self.show_thread.start()

    def setImgLabel(self, img_path, label, size=(100, 100)):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, size)
        h,w,d = img.shape
        qimg = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        qimg = QImage(qimg.data, w, h, w*d, QImage.Format_RGBA8888)
        label.setPixmap(QPixmap.fromImage(qimg))

    def resetImgLabel(self, label):
        self.setImgLabel(self.icon_file, label)

    def quit(self):
        self.setStreamer(False)
        QtCore.QCoreApplication.instance().quit()
    