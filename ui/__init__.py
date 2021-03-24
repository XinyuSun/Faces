import sys
import cv2
import os
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, 
                             QLineEdit, QMessageBox, QLabel, QMainWindow, 
                             QAction, QHeaderView)
from PyQt5.QtGui import QImage, QPixmap, QIcon, QStandardItemModel, QStandardItem, QWheelEvent
from PyQt5.QtCore import QModelIndex
from PyQt5 import QtCore, QtWidgets, QtGui
from utils.camera import cameraThread
from utils.face import detectionThread, compareThread, showThread
from ui.Ui_mainWindow import Ui_MainWindow
from ui.Ui_formWindow import Ui_Form
from copy import deepcopy
import pickle
import hashlib
import time
import math


def getHashCode(input: str) -> str:
    s = hashlib.sha256()
    s.update(input.encode())
    return s.hexdigest()


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
    
        self.form1 = uiForm()

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
        self.ui.menu1.triggered[QAction].connect(self.menuBarTriggered)

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
            roi = self.selected_face['roi']
            encodings = [self.selected_face['encoding']]
            name = self.ui.line1.text()
            print('save: %s'%name)
            savedict = {'name': name, 'encodings': encodings, 'roi': roi,
                        'time': time.strftime('%Y/%m/%d', time.localtime())}
            if not os.path.exists('./data'):
                os.makedirs('./data')
            
            hash_name = getHashCode(name)

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

    def menuBarTriggered(self, action: QAction):
        if action.text() == '所有用户':
            self.form1.loadSavedDatas()
            self.form1.updatePage(0)
            self.form1.show()

    def quit(self):
        self.setStreamer(False)
        QtCore.QCoreApplication.instance().quit()
    

class uiForm(QWidget):
    
    def __init__(self):
        super().__init__()

        self.model = QStandardItemModel()

        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.total_pages = 1
        self.cur_page = 0
        self.wheel_delta = 0.0
        self.units_in_page = 4

        self.units = [userUnit(self, 0, 20, 60), userUnit(self, 1, 210, 60), 
                     userUnit(self, 2, 20, 160), userUnit(self, 3, 210, 160)]

        self.saved_pkls = None
        self.saved_users_names = None
        self.show_users_pkls = None
        self.selected_unit_idx = None

        self.ui.lineEdit.textEdited[str].connect(self.searchUser)
        self.ui.button1.clicked.connect(self.removeUser)

        self.loadSavedDatas()
        self.updatePage(0)

    def loadSavedDatas(self):
        self.saved_users_names = []
        self.saved_pkls = os.listdir('./data/')
        self.total_pages = math.ceil(len(self.saved_pkls) / float(self.units_in_page))

        for pkl_file in self.saved_pkls:
            with open('./data/' + pkl_file, 'rb') as fo:
                data = pickle.load(fo)
                self.saved_users_names.append(data['name'])
        
        self.show_users_pkls = self.saved_pkls

    def searchUser(self, content):
        self.show_users_pkls = []

        for idx, name in enumerate(self.saved_users_names):
            if content in name:
                self.show_users_pkls.append(self.saved_pkls[idx])

        self.updatePage(0)

    def updatePage(self, page: int):
        saved_pkls = self.show_users_pkls
        start_idx = page * self.units_in_page
        self.total_pages = math.ceil(len(self.saved_pkls) / float(self.units_in_page))

        unit = {}

        if start_idx >= len(saved_pkls) or page < 0:
            return 1

        self.ui.progressBar.setMinimum(0)
        self.ui.progressBar.setMaximum(self.total_pages - 1)
        self.ui.progressBar.setValue(page)

        for idx in range(start_idx, start_idx + 4):
            if idx >= len(saved_pkls):
                self.units[idx % 4].hide()
                continue
            with open('./data/' + saved_pkls[idx], 'rb') as fo:
                data = pickle.load(fo)
                unit = deepcopy(data)

            self.units[idx % 4].update(unit['name'], unit['roi'], '2021/3/19', 'SCUT', idx)
        
        return 0

    def wheelEvent(self, event: QWheelEvent):
        self.wheel_delta += (float(event.angleDelta().y()) / 50)

        if int(self.wheel_delta) > 0:
            if self.updatePage(self.cur_page + 1) == 0:
                self.cur_page += 1
            self.wheel_delta = 0.0

        elif int(self.wheel_delta) < 0:
            if self.updatePage(self.cur_page - 1) == 0:
                self.cur_page -= 1
            self.wheel_delta = 0.0

    def selectUnit(self, selected):
        self.selected_unit_idx = selected

        for idx, unit in enumerate(self.units):
            if idx != selected:
                unit.setSelected(False)

    def removeUser(self):
        ret = QMessageBox.warning(self, 'Warning', '确认删除', QMessageBox.Ok | QMessageBox.No)
        if ret != QMessageBox.Ok:
            return

        if self.selected_unit_idx != None:
            pkl_file_to_rm = self.saved_pkls[self.selected_unit_idx]
            os.system('rm ./data/{}'.format(pkl_file_to_rm))
        
        self.loadSavedDatas()
        self.updatePage(0)


class userUnit(QWidget):
    unitSelectedSignal = QtCore.pyqtSignal(int)

    def __init__(self, form, idx, x, y):
        super().__init__(form)
        self.setGeometry(QtCore.QRect(x, y, 191, 81))
        self.setObjectName("widget")
        self.date = QtWidgets.QLabel(self)
        self.date.setGeometry(QtCore.QRect(90, 30, 60, 20))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.date.setFont(font)
        self.date.setObjectName("date")
        self.name = QtWidgets.QLabel(self)
        self.name.setGeometry(QtCore.QRect(90, 6, 60, 20))
        self.name.setObjectName("name")
        self.profile = QtWidgets.QLabel(self)
        self.profile.setGeometry(QtCore.QRect(0, 0, 80, 80))
        self.profile.setStyleSheet("background-color: rgb(220, 217, 222);")
        self.profile.setObjectName("profile")
        self.remark = QtWidgets.QLabel(self)
        self.remark.setGeometry(QtCore.QRect(90, 45, 60, 20))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.remark.setFont(font)
        self.remark.setObjectName("remark")
        self.hl = QtWidgets.QFrame(self)
        self.hl.setGeometry(QtCore.QRect(90, 20, 61, 21))
        self.hl.setFrameShape(QtWidgets.QFrame.HLine)
        self.hl.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.hl.setObjectName("hl")

        self._translate = QtCore.QCoreApplication.translate
        self.date.setText(self._translate("Form", "2020/7/8"))
        self.name.setText(self._translate("Form", "User 1"))
        self.profile.setText(self._translate("Form", "TextLabel"))
        self.remark.setText(self._translate("Form", "SCUT"))

        self.vl = QtWidgets.QFrame(self)
        self.vl.setGeometry(QtCore.QRect(170, 10, 3, 61))
        self.vl.setFrameShape(QtWidgets.QFrame.VLine)
        self.vl.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.vl.setObjectName("vl")
        self.vl.setStyleSheet("background-color: rgb(120, 195, 255);")
        self.vl.hide()

        self._isSelected = False
        self.unitID = idx
        self.userID = None

        self.setMouseTracking(True)
        self.unitSelectedSignal[int].connect(form.selectUnit)

    def update(self, name=None, profile=None, date=None, remark=None, userid=None):
        self.date.setText(self._translate("Form", date))
        self.name.setText(self._translate("Form", name))
        self.remark.setText(self._translate("Form", remark))
        self.userID = userid

        if profile is not None:
            img = cv2.resize(profile, (80, 80))
            h,w,d = img.shape
            qimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimg = QImage(qimg.data, w, h, w*d, QImage.Format_RGB888)
            self.profile.setPixmap(QPixmap.fromImage(qimg))
        
        self.vl.hide()
        self._isSelected = False
        self.show()

    def mousePressEvent(self, event):
        if self.isSelected():
            self.setSelected(False)
        else:
            self.setSelected(True)
            self.unitSelectedSignal.emit(self.userID)

    def isSelected(self):
        return self._isSelected

    def setSelected(self, stat):
        if stat:
            self._isSelected = True
            self.vl.show()
        else:
            self._isSelected = False
            self.vl.hide()