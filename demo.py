import face_recognition
import cv2
import numpy as np
from utils.log import setupLogger
from utils.ui import uiWindow
import sys
from PyQt5.QtWidgets import QApplication


KEY_MASK = 0xff


def main():
    loger = setupLogger()
    vstreamer = cv2.VideoCapture()
    vstreamer.open(0)
    if not vstreamer.isOpened():
        logger.error('camera open failed')

    while True:
        key_code = cv2.waitKey(3) & KEY_MASK
        ret, image = vstreamer.read()
        if not ret:
            logger.error('capture frame failed')
        locs = face_recognition.face_locations(image)
        encodings = face_recognition.face_encodings(image, locs)

        for loc in locs:
            t,r,b,l = loc
            image = cv2.rectangle(image, (l,t), (r,b), (255, 255, 0), 2)

        if key_code == ord('q'):
            break
        elif key_code == ord(' '):
            save_encoding = encodings


        cv2.imshow('frame', image)


if __name__ == '__main__':
    # main()
    app = QApplication(sys.argv)
    window = uiWindow()
    sys.exit(app.exec_())