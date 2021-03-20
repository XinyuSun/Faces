import cv2
import os
import pickle
import numpy as np
from PyQt5.QtCore import QThread
from PyQt5.QtGui import QImage, QPixmap
from copy import deepcopy
import face_recognition
from multiprocessing import Process, Queue, Manager
import multiprocessing as mp
from PyQt5 import QtCore
import time

face_tolerance = 0.45

def detection(img_queue, det_queue, exit_signal_queue):
    while True:
        if exit_signal_queue.empty():
            print('process exit')
            return
        else:
            exit_signal_queue.get(block=False)
        
        if not img_queue.empty():
            cam_frame = img_queue.get()
            locs = face_recognition.face_locations(cam_frame)
            encodings = face_recognition.face_encodings(cam_frame, locs, 5, "large")
            rois = [cam_frame[t:b,l:r,:] for (t,r,b,l) in locs]
            det_queue.put({'locs': locs, 'encodings': encodings, 'rois': deepcopy(rois)})
        else:
            cv2.waitKey(100)


class detectionThread(QThread):
    
    def __init__(self, widget):
        super().__init__()

        self.related_widget = widget

        self.detp = Process(target=detection, args=(
            self.related_widget.camera_thread.img_queue,
            self.related_widget.camera_thread.det_queue,
            self.related_widget.camera_thread.exit_signal_queue,
        ))

    def run(self):
        self.detp.start()
        self.detp.join()


class compareThread(QThread):

    def __init__(self, widget):
        super().__init__()

        self.related_widget = widget

    def run(self):
        saved_faces = os.listdir('./data')
        saved_faces = [face for face in saved_faces if face[-4:] == '.pkl']
        saved_names = []
        saved_encodings = []
        paired_face = []
        target_face_encodings = self.related_widget.current_faces['encodings']

        if len(saved_faces):
            for face in saved_faces:
                with open('./data/' + face, 'rb') as fo:
                    data = pickle.load(fo)
                    for encoding in data['encodings']:
                        saved_names.append(data['name'])
                        saved_encodings.append(encoding)
            
            for idx, encoding in enumerate(target_face_encodings):
                scores = face_recognition.face_distance(saved_encodings, encoding)
                pair_idx = np.argmin(scores)
                pair_score = scores[pair_idx]
                if pair_score >= face_tolerance:
                    pair_name = 'Unpaired'
                else:
                    pair_name = saved_names[pair_idx]
                
                self.related_widget.current_faces['names'][idx] = pair_name
                self.related_widget.current_faces['scores'][idx] = pair_score
                paired_face.append(pair_name)
        else:
            self.related_widget.current_faces['names'] = ['Unpaired' for i in range(len(target_face_encodings))]
            self.related_widget.current_faces['scores'] = [1.0 for i in range(len(target_face_encodings))]

        '''
        if len(paired_face):
            self.related_widget.ui.line1.setText(paired_face[0])
        else:
            self.related_widget.ui.line1.setText('Unpaired')
        '''

        self.related_widget.update_results_signal.emit()


class showThread(QThread):
    
    def __init__(self, widget):
        super().__init__()

        self.related_widget = widget

    def run(self):
        current_faces_dict = self.related_widget.current_faces
        name_list = [face['name'] for face in self.related_widget.cache_faces]
        encoding_list = [face['encoding'] for face in self.related_widget.cache_faces]
        sl_face = self.related_widget.selected_face

        for face_idx in range(len(current_faces_dict['names'])):

            t,r,b,l = current_faces_dict['locs'][face_idx]

            cu_face = {'name': current_faces_dict['names'][face_idx],
                       'loc': current_faces_dict['locs'][face_idx],
                       'encoding': current_faces_dict['encodings'][face_idx],
                       'roi': deepcopy(current_faces_dict['rois'][face_idx]),
                       'time': time.time()}

            store_face_sig = True

            if cu_face['name'] in name_list and cu_face['name'] != 'Unpaired':
                continue
            elif len(encoding_list):
                dist = face_recognition.face_distance(cu_face['encoding'], encoding_list)
                if dist.min() < face_tolerance:
                    store_face_sig = False

            if store_face_sig:
                self.related_widget.cache_faces.append(cu_face)

        # self.related_widget.ui.line2.setText('{} face'.format(len(self.related_widget.cache_faces)))

        # kick out of queue
        for idx, face in enumerate(self.related_widget.cache_faces):
            if sl_face != None and face['name'] == sl_face['name']:
                # print('pass: %s'%face['name'])
                continue
            # time out
            if time.time() - face['time'] > 5.0:
                self.related_widget.cache_faces.remove(self.related_widget.cache_faces[idx])

        while len(self.related_widget.cache_faces) > 4:
            rm_idx = 0
            if sl_face != None and self.related_widget.cache_faces[0]['encoding'].any() == sl_face['encoding'].any():
                rm_idx = 1

            self.related_widget.cache_faces.remove(self.related_widget.cache_faces[rm_idx])

        for idx, face in enumerate(self.related_widget.cache_faces):
            roi = cv2.resize(face['roi'], (100, 100))
            w,h,d = roi.shape
            qimg = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            qimg = QImage(qimg.data, w, h, w * d, QImage.Format_RGB888)
            self.related_widget.roiLabel[idx].setPixmap(QPixmap.fromImage(qimg))
            self.related_widget.names[idx].setText(face['name'])

        # fill the label with png file
        for idx in range(len(self.related_widget.cache_faces), 4):
            self.related_widget.resetImgLabel(self.related_widget.roiLabel[idx])
            self.related_widget.names[idx].setText('User {}'.format(idx + 1))