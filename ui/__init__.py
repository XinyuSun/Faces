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


class uiWindow(QMainWindow):
    # signal
    found_face_signal = QtCore.pyqtSignal()
    update_results_signal = QtCore.pyqtSignal()
    face_selected_signal = QtCore.pyqtSignal(int)

    def __init__(self):
        super().__init__()

        self.icon_file = 'ui/resources/icon1.png'
        self.mouse_pos = None

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.roiLabel = [self.ui.roiLabel1, self.ui.roiLabel2, self.ui.roiLabel3, self.ui.roiLabel4]
        self.names = [self.ui.name1, self.ui.name2, self.ui.name3, self.ui.name4]
        for label in self.roiLabel:
            self.setImgLabel(self.icon_file, label)

        # connection
        self.ui.button1.clicked.connect(self.saveFace)
        self.ui.button2.clicked.connect(self.quit)
        self.found_face_signal.connect(self.compareFace)
        self.update_results_signal.connect(self.showFace)
        self.face_selected_signal.connect(self.selectFace)

        # thread
        self.camera_thread = cameraThread(self)
        self.detection_thread = detectionThread(self)
        self.compare_thread = compareThread(self)
        self.show_thread = showThread(self)

        # current faces
        self.current_faces = {'names': None, 'locs': None, 'encodings': None, 'scores': None}
        self.cache_faces = []
        self.selected_face = None

        self.setStreamer(start=True)
        self.setMouseTracking(True)
        self.show()

    def update(self, image):
        h, w, d = image.shape
        self.resize(w + 20, h + 20 + 40)
        self.ui.imgLabel.resize(w, h)
        qimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        qimg = QImage(qimg.data, w, h, w * d, QImage.Format_RGB888)
        self.ui.imgLabel.setPixmap(QPixmap.fromImage(qimg))

    def setStreamer(self, start=False):
        if start:
            self.camera_thread.start()
            self.detection_thread.start()
        else:
            self.camera_thread.close()
            self.close()

    def saveFace(self):
        if self.ui.line1.text() == '' or self.ui.line1.text() == 'Unpaired':
            QMessageBox.warning(self, 'Warning', '请输入姓名', QMessageBox.Ok)
            return 
            
        if len(self.camera_thread.dets) > 0:
            encodings = [self.selected_face['encoding']]
            name = self.ui.line1.text()
            print('save: %s'%name)
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

    def selectFace(self, idx):
        if idx >= len(self.cache_faces):
            self.selected_face = None
            return
        self.selected_face = self.cache_faces[idx]
        self.ui.line1.setText(self.selected_face['name'])

    def setImgLabel(self, img_path, label, size=(100, 100)):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, size)
        h,w,d = img.shape
        qimg = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        qimg = QImage(qimg.data, w, h, w*d, QImage.Format_RGBA8888)
        label.setPixmap(QPixmap.fromImage(qimg))

    def resetImgLabel(self, label):
        self.setImgLabel(self.icon_file, label)

    def mousePressEvent(self, event):
        self.mouse_pos = event.windowPos()
        x,y = self.mouse_pos.x(), self.mouse_pos.y()
        print('({},{})'.format(self.mouse_pos.x(), self.mouse_pos.y()))
        
        if x > 1110 and x < 1210 and y > 10 and y < 530:
            idx = (y - 10) // 130
            self.face_selected_signal.emit(idx)
        else:
            self.selected_face = None

    def quit(self):
        self.setStreamer(False)
        QtCore.QCoreApplication.instance().quit()
    