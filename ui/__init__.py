import sys
import cv2
import os
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QLineEdit, QMessageBox, QLabel, QMainWindow
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5 import QtCore
from utils.camera import cameraThread
from utils.face import detectionThread, compareThread
from ui.Ui_mainWindow import Ui_MainWindow
import pickle
import hashlib
import time


FONT_COLOR = '#ffffff'
BACKGROUND_COLOR = '#1e1e1e'
BUTTON_COLOR = '#333333'
BUTTON_COLOR_HOVER = '#252525'
LINE_COLOR = '#333333'

class uiWindow(Ui_MainWindow, QWidget):
    # signal
    found_face_signal = QtCore.pyqtSignal()
    update_results_signal = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()

        icon_file = 'ui/resources/icon1.png'

        self.mainWindow = QMainWindow()
        self.setupUi(self.mainWindow)

        self.roiLabel = [self.roiLabel1, self.roiLabel2, self.roiLabel3, self.roiLabel4]
        for label in self.roiLabel:
            self.setImgLabel(icon_file, label)

        # connection
        self.button1.clicked.connect(self.saveFace)
        self.button2.clicked.connect(self.quit)
        self.found_face_signal.connect(self.compareFace)
        # self.update_results_signal.connect(self.showFace)

        # thread
        self.camera_thread = cameraThread(self)
        self.detection_thread = detectionThread(self)
        self.compare_thread = compareThread(self)

        # current faces
        self.current_faces = {'names': None, 'locs': None, 'encodings': None, 'scores': None}
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

    def showFace(self):
        # update per 1s
        time_gap = time.time() - self.last_show_time
        if time_gap < 2.0:
            return
        else:
            self.last_show_time = time.time()

        # sorted by compare score
        idxs = np.argsort(np.array(self.current_faces['scores']))
        for key in self.current_faces.keys():
            np_list = np.array(self.current_faces[key])
            self.current_faces[key] = list(np_list[idxs])
        face_to_show = []

        # get top 4 faces
        for i in range(4 if idxs.size >= 4 else idxs.size):
            face_to_show.append({k:self.current_faces[k][i] for k in self.current_faces})

        for i, face in enumerate(face_to_show):
            t,r,b,l = face['locs']
            roi_rigion = self.camera_thread.last_frame[t:b,l:r,:]

            # resize face roi to 100 x 100
            roi_rigion = cv2.resize(roi_rigion, (100, 100))
            w,h,d = roi_rigion.shape
            qimg = cv2.cvtColor(roi_rigion, cv2.COLOR_BGR2RGB)
            qimg = QImage(qimg.data, w, h, w * d, QImage.Format_RGB888)
            self.roiLabel[i].setPixmap(QPixmap.fromImage(qimg))
            self.roiLabel[i].raise_()

    def setImgLabel(self, img_path, label, size=(100, 100)):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, size)
        h,w,d = img.shape
        qimg = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        qimg = QImage(qimg.data, w, h, w*d, QImage.Format_RGBA8888)
        label.setPixmap(QPixmap.fromImage(qimg))

    def quit(self):
        self.setStreamer(False)
        QtCore.QCoreApplication.instance().quit()
    

class _uiWindow(QWidget):
    # signal
    found_face_signal = QtCore.pyqtSignal()
    update_results_signal = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()

        self.initUI()
        self.isTerminated = False

    def initUI(self):
        self.setGeometry(50, 50, 1280, 720)
        # self.resize(640, 480)
        self.setWindowTitle('face')
        self.setObjectName('mainWindow')
        self.setStyleSheet('QWidget{background-color:%s}'%BACKGROUND_COLOR)

        self.imgLabel = QLabel(self)
        self.imgLabel.setGeometry(10, 10, 1280, 720)

        self.roiLabel = [QLabel(self) for i in range(4)]

        self.button1 = QPushButton(self)
        self.button1.setText('save')
        self.button1.setGeometry(1120, 740, 80, 30)
        self.button1.setStyleSheet('QPushButton{background:%s;border-radius:5px;color:%s}\
                                       QPushButton:hover{background:%s;}'%(BUTTON_COLOR, FONT_COLOR, BUTTON_COLOR_HOVER))

        self.button2 = QPushButton(self)
        self.button2.setText('quit')
        self.button2.setGeometry(1210, 740, 80, 30)
        self.button2.setStyleSheet('QPushButton{background:%s;border-radius:5px;color:%s}\
                                       QPushButton:hover{background:%s;}'%(BUTTON_COLOR, FONT_COLOR, BUTTON_COLOR_HOVER))

        self.line1 = QLineEdit(self)
        self.line1.setGeometry(100+80+10, 10+720+15, 160, 20)
        self.line1.setText('Msgs')
        self.line1.setReadOnly(True)
        self.line1.setStyleSheet('QLineEdit{background:%s;color:%s;border-color:%s}'%(LINE_COLOR, FONT_COLOR, LINE_COLOR))

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
        self.update_results_signal.connect(self.showFace)

        # thread
        self.camera_thread = cameraThread(self)
        self.detection_thread = detectionThread(self)
        self.compare_thread = compareThread(self)

        # current faces
        self.current_faces = {'names': None, 'locs': None, 'encodings': None, 'scores': None}
        self.last_show_time = 0.0

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

    def showFace(self):
        # update per 1s
        time_gap = time.time() - self.last_show_time
        if time_gap < 2.0:
            return
        else:
            self.last_show_time = time.time()

        # sorted by compare score
        idxs = np.argsort(np.array(self.current_faces['scores']))
        for key in self.current_faces.keys():
            np_list = np.array(self.current_faces[key])
            self.current_faces[key] = list(np_list[idxs])
        face_to_show = []

        # get top 4 faces
        for i in range(4 if idxs.size >= 4 else idxs.size):
            face_to_show.append({k:self.current_faces[k][i] for k in self.current_faces})

        for i, face in enumerate(face_to_show):
            t,r,b,l = face['locs']
            roi_rigion = self.camera_thread.last_frame[t:b,l:r,:]

            # resize face roi to 130 x 130
            roi_rigion = cv2.resize(roi_rigion, (130, 130))
            w,h,d = roi_rigion.shape
            self.roiLabel[i].setGeometry(10+1280-w, 10+(h+10)*i, w, h)
            qimg = cv2.cvtColor(roi_rigion, cv2.COLOR_BGR2RGB)
            qimg = QImage(qimg.data, w, h, w * d, QImage.Format_RGB888)
            self.roiLabel[i].setPixmap(QPixmap.fromImage(qimg))
            self.roiLabel[i].raise_()
            

    def quit(self):
        self.setStreamer(False)
        QtCore.QCoreApplication.instance().quit()
