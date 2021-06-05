import face_recognition
import cv2
import numpy as np
import copy

def detect_pupil(frame, rois, landmarks):

    eye_score = []

    for i, landmark in enumerate(landmarks):
        le_list = landmark['left_eye']
        re_list = landmark['right_eye']
        #print(le_list)
        le_cnt = np.array(le_list)
        le_cnt.resize(len(le_list), 1, 2)
        re_cnt = np.array(re_list)
        re_cnt.resize(len(re_list), 1, 2)
        #print(le_cnt)

        #le_rect = np.int0(cv2.boxPoints(cv2.minAreaRect(le_cnt)))
        le_rect = cv2.boundingRect(le_cnt)
        
        l,t,w,h = le_rect
        r,b = l+w,t+h
        le_roi = copy.deepcopy(frame[int(t):int(b),int(l):int(r),:])

        cv2.imshow('le', le_roi)

        le_roi = cv2.cvtColor(le_roi, cv2.COLOR_BGR2GRAY)
        ret, le_roi_thresh = cv2.threshold(le_roi, 20, 255, cv2.THRESH_OTSU)
        ret, le_roi_thresh = cv2.threshold(le_roi_thresh, 5, 255, cv2.THRESH_BINARY_INV)


        kernel = np.ones((3, 3), np.uint8)
        le_roi_thresh = cv2.erode(le_roi_thresh, kernel, iterations=1)
        le_roi_thresh = cv2.dilate(le_roi_thresh, kernel, iterations=1)

        cv2.imshow('le_thresh', le_roi_thresh)

        hist = cv2.calcHist([le_roi_thresh], [0], None, [256], [0,256])
        
        score = 0.5 * w / h + 0.5 * hist[0] / hist[255]
        print(score)

        cv2.rectangle(frame, (l,t), (r,b), (0,255,0), 3)
        #cv2.drawContours(frame, [le_rect], -1, (0, 255, 0), 2)



if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    while cv2.waitKey(3) & 0xff != ord('q'):
        ret, frame = cap.read()
        locs = face_recognition.face_locations(frame)
        rois = [frame[t:b,l:r,:] for (t,r,b,l) in locs]
        landmarks = face_recognition.face_landmarks(frame)
        detect_pupil(frame, rois, landmarks)
        cv2.imshow('frame', frame)
    
    cap.release()