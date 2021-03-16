import cv2
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
            det_queue.put({'locs': locs})
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
