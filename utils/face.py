import cv2
import os
import pickle
import numpy as np
from PyQt5.QtCore import QThread
from copy import deepcopy
import face_recognition
from multiprocessing import Process, Queue, Manager
from PyQt5 import QtCore


def detection(img_queue, det_queue, exit_signal_queue):
    while True:
        if exit_signal_queue.empty():
            break
        else:
            exit_signal_queue.get(block=False)
        
        if not img_queue.empty():
            cam_frame = img_queue.get()
            locs = face_recognition.face_locations(cam_frame)
            encodings = face_recognition.face_encodings(cam_frame, locs)
            det_queue.put({'locs': locs, 'encodings': encodings})
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
        if len(saved_faces) == 0:
            return

        for face in saved_faces:
            with open('./data/' + face, 'rb') as fo:
                data = pickle.load(fo)
                for encoding in data['encodings']:
                    saved_names.append(data['name'])
                    saved_encodings.append(encoding)
        
        target_face_encodings = self.related_widget.current_faces['encodings']
        paired_face = []
        for idx, encoding in enumerate(target_face_encodings):
            scores = face_recognition.face_distance(saved_encodings, encoding)
            pair_idx = np.argmax(scores)
            pair_score = scores[pair_idx]
            if pair_score >= 0.6:
                pair_name = 'Unpaired'
            else:
                pair_name = saved_names[pair_idx]
            
            self.related_widget.current_faces['names'][idx] = pair_name
            self.related_widget.current_faces['scores'][idx] = pair_score
            paired_face.append(pair_name)

        if len(paired_face):
            self.related_widget.line1.setText(paired_face[0])
        else:
            self.related_widget.line1.setText('Unpaired')

        self.related_widget.update_results_signal.emit()


            