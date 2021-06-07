from utils.eyenet import eyeNet, preprocess
import face_recognition
import cv2
import numpy as np
import copy
import math
import time
import torch
from PIL import Image


def order_points_new(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    if leftMost[0,1]!=leftMost[1,1]:
        leftMost=leftMost[np.argsort(leftMost[:,1]),:]
    else:
        leftMost=leftMost[np.argsort(leftMost[:,0])[::-1],:]
    (tl, bl) = leftMost
    if rightMost[0,1]!=rightMost[1,1]:
        rightMost=rightMost[np.argsort(rightMost[:,1]),:]
    else:
        rightMost=rightMost[np.argsort(rightMost[:,0])[::-1],:]
    (tr,br)=rightMost
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


def drow_box(img, cnt):
    rotated_box = cv2.minAreaRect(cnt)

    box = cv2.boxPoints(rotated_box)
    box = np.int0(box)

    vertices = order_points_new(box)

    '''
    colors = [(255,0,0),(0,255,0),(0,0,255),(50,255,255)]

    for i, (x, y) in enumerate(vertices):
        cv2.circle(img, (int(x), int(y)), 1, colors[i], 2)
    '''

    return img, rotated_box, vertices


def crop(img, cnt):
    img, rotated_box, vertices = drow_box(img, cnt)

    width = int(rotated_box[1][0])
    height = int(rotated_box[1][1])

    if width > height:
        w = width
        h = height
    else:
        w = height
        h = width

    src_pts = vertices.astype("float32")
    dst_pts = np.array([[0, 0],
                        [w - 1, 0],
                        [w - 1, h - 1],
                        [0, h - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (w, h))

    return warped


def enhance_roi(frame, cnt):
    roi = crop(frame, cnt)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.GaussianBlur(roi, (3, 3), 0)
    roi = cv2.equalizeHist(roi)
    return roi


def cal_distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)


def cal_ear_score(pts):
    return (cal_distance(pts[1],pts[5]) + cal_distance(pts[2],pts[4])) / (2*cal_distance(pts[0],pts[3]))


def detect_pupil(model, frame, landmarks):

    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255)]
    eye_scores = []
    eye_status = []

    for i, landmark in enumerate(landmarks):
        le_list = landmark['left_eye']
        re_list = landmark['right_eye']
        #print(le_list)
        le_cnt = np.array(le_list)
        le_cnt.resize(len(le_list), 1, 2)
        re_cnt = np.array(re_list)
        re_cnt.resize(len(re_list), 1, 2)

        #for i, pt in enumerate(le_list):
        #    cv2.circle(frame, pt, 2, colors[i], 3)

        re_roi = enhance_roi(frame, re_cnt)
        le_roi = enhance_roi(frame, le_cnt)

        #cv2.imshow('le', le_roi)
        cv2.imwrite(f'dataset/open/eye_{str(int(time.time() * 100))}.png', le_roi)
        cv2.imwrite(f'dataset/open/eye_{str(int(time.time() * 110))}.png', re_roi)

        if model is not None:
            le_pil_image = Image.fromarray(le_roi).convert('RGB')
            re_pil_image = Image.fromarray(re_roi).convert('RGB')        
            le_input = preprocess(le_pil_image)
            re_input = preprocess(re_pil_image)
            input = torch.stack([le_input, re_input])
            res = model.inference(input).cpu().numpy()
        else:
            res = 0
        eye_status.append(res)

        score = 0.5 * cal_ear_score(re_list) + 0.5 * cal_ear_score(le_list)
        eye_scores.append(score)

        print('score:%.3f, res:%s'%(score, str(res)))

    return eye_scores, eye_status


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    model = eyeNet('model_best.pth')

    while cv2.waitKey(3) & 0xff != ord('q'):
        ret, frame = cap.read()
        locs = face_recognition.face_locations(frame)
        landmarks = face_recognition.face_landmarks(frame)
        detect_pupil(model, frame, landmarks)
        #detect_pupil(None, frame, landmarks)
        cv2.imshow('frame', frame)
    
    cap.release()