import cv2
from PyQt5.QtCore import QThread
from copy import deepcopy
import face_recognition
from multiprocessing import Process, Queue, Manager
from PyQt5 import QtCore


class cameraThread(QThread):
    def __init__(self, widget):
        
        super().__init__()
        self.vstreamer = cv2.VideoCapture(0)
        self.frame = None
        self.related_widget = widget
        self.update_cache = True

        self.stop_signal = False

        self.img_queue = Queue(maxsize=1)
        self.det_queue = Queue()
        self.exit_signal_queue = Queue(maxsize=10)

    def close(self):
        self.stop_signal = True

    def run(self):
        while True:
            if self.stop_signal:
                self.vstreamer.release()
                return

            ret, frame = self.vstreamer.read()

            # horizontal flip
            frame = cv2.flip(frame, 1)
            if self.related_widget.isActiveWindow():
                self.related_widget.update(frame)

            # cache
            if self.img_queue.empty():
                self.frame = deepcopy(frame)
                self.img_queue.put(self.frame)
                self.related_widget.line.setText('put image')

            # feed
            if not self.exit_signal_queue.full():
                self.exit_signal_queue.put(-1, block=False)

            # get detection results
            if not self.det_queue.empty():
                dets = self.det_queue.get()
                self.related_widget.line.setText('{}'.format(dets['locs']))

            cv2.waitKey(1)
    