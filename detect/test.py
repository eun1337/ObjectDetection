import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os
from darknet import Darknet
import pickle as pkl
import random
import dlib
import winsound

# 졸음 인식 설정
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
EYES = list(range(36, 48))

frame_width = 640
frame_height = 480

title_name = 'Drowsiness and Object Detection'

face_cascade_name = './haarcascade_frontalface_alt.xml'  # -- 본인 환경에 맞게 변경할 것
face_cascade = cv2.CascadeClassifier()
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)

predictor_file = './shape_predictor_68_face_landmarks.dat'  # -- 본인 환경에 맞게 변경할 것
predictor = dlib.shape_predictor(predictor_file)

status = 'Awake'
number_closed = 0
min_EAR = 0.25
closed_limit = 10  # -- 눈 감김이 10번 이상일 경우 졸음으로 간주
show_frame = None
sign = None
color = (0, 255, 0)

def getEAR(points):
    A = np.linalg.norm(points[1] - points[5])
    B = np.linalg.norm(points[2] - points[4])
    C = np.linalg.norm(points[0] - points[3])
    return (A + B) / (2.0 * C)

def detect_drowsiness(image):
    global number_closed
    global color
    global show_frame
    global sign
    global status

    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    faces = face_cascade.detectMultiScale(frame_gray)
    
    if len(faces) == 0:
        cv2.putText(show_frame, "No face detected", (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return
    
    for (x, y, w, h) in faces:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        points = np.matrix([[p.x, p.y] for p in predictor(image, rect).parts()])
        show_parts = points[EYES]
        right_eye_EAR = getEAR(points[RIGHT_EYE])
        left_eye_EAR = getEAR(points[LEFT_EYE])
        mean_eye_EAR = (right_eye_EAR + left_eye_EAR) / 2 

        if mean_eye_EAR > min_EAR:
            color = (0, 255, 0)
            status = 'Awake'
            number_closed = number_closed - 1
            if number_closed < 0:
                number_closed = 0
        else:
            color = (0, 0, 255)
            status = 'Sleep'
            number_closed = number_closed + 1
                     
        sign = 'Sleep count : ' + str(number_closed) + ' / ' + str(closed_limit)

        # 졸음 확정시 알람 설정
        if number_closed > closed_limit:
            winsound.PlaySound("./alarm.wav", winsound.SND_FILENAME)  # -- 본인 환경에 맞게 변경할 것
        
    cv2.putText(show_frame, status, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
    cv2.putText(show_frame, sign, (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def arg_parse():
    """
    인수 파싱 함수
    """
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file", default="yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile", default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed", default="416", type=str)

    return parser.parse_args()

def prep_image(img, inp_dim):
    """
    이미지 크기를 YOLO 모델의 입력 크기에 맞게 조정하고, 필요한 전처리를 수행합니다.
    """
    orig_im = img.copy()
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

args = arg_parse()
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
CUDA = torch.cuda.is_available()

num_classes = 80
classes = load_classes("coco.names")  # YOLO 클래스 이름 파일 경로

# 신경망 설정
print("네트워크 로드 중.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("네트워크가 성공적으로 로드되었습니다")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32

# GPU가 사용 가능하면 모델을 GPU에 올립니다
if CUDA:
    model.cuda()

# 모델을 평가 모드로 설정
model.eval()

# 경계 상자 색상 로드
colors = pkl.load(open("pallete", "rb"))

# 웹캠 열기
cap = cv2.VideoCapture(0)
time.sleep(2.0)
if not cap.isOpened():
    print('Could not open video')
    exit(0)

def write(x, results):
    c1 = tuple(map(int, x[1:3]))
    c2 = tuple(map(int, x[3:5]))
    img = results
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("웹캠에서 프레임을 읽을 수 없습니다.")
        break

    img, orig_im, dim = prep_image(frame, inp_dim)
    im_dim = torch.FloatTensor(dim).repeat(1, 2)

    if CUDA:
        im_dim = im_dim.cuda()
        img = img.cuda()

    with torch.no_grad():
        output = model(Variable(img), CUDA)
    output = write_results(output, confidence, num_classes, nms_conf=nms_thesh)

    if type(output) != int:
        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(inp_dim / im_dim, 1)[0].view(-1, 1)

        output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
        output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2
        output[:, 1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

        for obj in output:
            cls = int(obj[-1])
            if classes[cls] in ['remote', 'cell phone']:
                winsound.PlaySound("./alarm.wav", winsound.SND_FILENAME)  # -- 본인 환경에 맞게 변경할 것

        list(map(lambda x: write(x, orig_im), output))

    show_frame = orig_im.copy()
    detect_drowsiness(orig_im)

    cv2.imshow(title_name, show_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
torch.cuda.empty_cache()
