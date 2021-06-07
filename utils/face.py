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
from utils.pupil import detect_pupil
from utils.eyenet import eyeNet

face_tolerance = 0.42

def detection(img_queue, det_queue, exit_signal_queue):
    if os.path.exists('model_best.pth'):
        model = eyeNet('model_best.pth')
    else:
        model = None

    while True:
        if exit_signal_queue.empty():
            print('process exit')
            return
        else:
            exit_signal_queue.get(block=False)
        
        if not img_queue.empty():
            tic = time.time()
            cam_frame = img_queue.get()

            locs = face_recognition.face_locations(cam_frame)
            encodings = face_recognition.face_encodings(cam_frame, locs, 5, "large")
            landmarks = face_recognition.face_landmarks(cam_frame)
            eye_scores, eye_status = detect_pupil(model, cam_frame, landmarks)

            rois = [cam_frame[t:b,l:r,:] for (t,r,b,l) in locs]
            det_queue.put({'locs': locs, 'encodings': encodings, \
                           'rois': deepcopy(rois), 'landmarks': landmarks, \
                           'eye_scores': eye_scores, 'eye_status': eye_status})

            toc = time.time() - tic
            print(f'use time: {toc * 1000}ms')
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
                       'eye_scores': current_faces_dict['eye_scores'][face_idx],
                       'eye_status': current_faces_dict['eye_status'][face_idx],
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

        # display face 
        for idx, face in enumerate(self.related_widget.cache_faces):
            # get face roi
            roi = cv2.resize(face['roi'], (100, 100))
            name_to_show = face['name']

            # display score
            cv2.putText(roi, ('%.2f'%face['eye_scores']), (5, 95), cv2.FONT_HERSHEY_COMPLEX, 0.5, (100,100,0), 2)
            # draw out a box if eye closed
            if face['eye_scores'] < 0.15 or np.all(face['eye_status'] == 0):
                cv2.rectangle(roi, (0,0), (99,99), (40,40,255), 5)
                name_to_show += '(走神)'

            # show face
            w,h,d = roi.shape
            qimg = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            qimg = QImage(qimg.data, w, h, w * d, QImage.Format_RGB888)
            self.related_widget.roiLabel[idx].setPixmap(QPixmap.fromImage(qimg))
            self.related_widget.names[idx].setText(name_to_show)

        # fill the label with png file
        for idx in range(len(self.related_widget.cache_faces), 4):
            self.related_widget.resetImgLabel(self.related_widget.roiLabel[idx])
            self.related_widget.names[idx].setText('User {}'.format(idx + 1))