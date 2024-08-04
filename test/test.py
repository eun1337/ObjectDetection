import time
import torch # pytorch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 # opencv
from util import *
import argparse
import os
from darknet import Darknet
import pickle as pkl
import random
import dlib # 얼굴인식 및 랜드마크 검출
import winsound # 경고음

# 운전자 얼굴 정보를 저장할 변수
driver_face = None

# 졸음 인식 설정
# 눈과 입의 랜드마크 인덱스 정의
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
EYES = list(range(36, 48))
MOUTH = list(range(48, 68))

frame_width = 640
frame_height = 480

title_name = 'Drowsiness and Object Detection'

# 얼굴 검출을 위한 Haar Cascade 경로 설정
face_cascade_name = './haarcascade_frontalface_alt.xml'  # -- 본인 환경에 맞게 변경할 것(상대 경로)
face_cascade = cv2.CascadeClassifier()
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
# 얼굴 랜드마크 검출을 위한 dlib 예측 모델 파일 경로 설정
# 얼굴 랜드마크 예측 모델
predictor_file = './shape_predictor_68_face_landmarks.dat'  # -- 본인 환경에 맞게 변경할 것(상대 경로)
predictor = dlib.shape_predictor(predictor_file)

status = 'Awake'
number_closed = 0
min_EAR = 0.25
closed_limit = 10  # -- 눈 감김이 10번 이상일 경우 졸음으로 간주
yawn_count = 0  # 하품 횟수 초기화
yawn_limit = 3  # 하품이 3번 감지되면 알람
show_frame = None
sign = None
color = (0, 255, 0)
last_alarm_time = 0
alarm_interval = 10  # 알람 사이의 최소 시간 간격 (초)

# 캘리브레이션 관련 변수
calibration_frames = 20 # 프레임 수 줄일수록 캘리브레이션 시간 줄어듦
calibration_counter = 0
total_EAR = 0
calibrated = False

# EAR (Eye Aspect Ratio) 계산 함수
# 눈의 랜드마크 간의 거리 계산
def getEAR(points):
    A = np.linalg.norm(points[1] - points[5])
    B = np.linalg.norm(points[2] - points[4])
    C = np.linalg.norm(points[0] - points[3])
    return (A + B) / (2.0 * C)

# MAR (Mouth Aspect Ratio) 계산 함수
# 입의 랜드마크 간의 거리 계산
def getMAR(points):
    A = np.linalg.norm(points[13] - points[19])
    B = np.linalg.norm(points[14] - points[18])
    C = np.linalg.norm(points[15] - points[17])
    D = np.linalg.norm(points[12] - points[16])
    return (A + B + C) / (2.0 * D)

# 졸음 감지
def detect_drowsiness(image):
    global driver_face 
    global number_closed
    global yawn_count
    global color
    global show_frame
    global sign
    global status
    global last_alarm_time
    global calibration_counter
    global total_EAR
    global calibrated
    global min_EAR

    # 이미지를 그레이스케일로 변환하고 히스토그램 평활화
    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    faces = face_cascade.detectMultiScale(frame_gray)
    
    # 얼굴 인식 실패 시 처리
    if len(faces) == 0:
        cv2.putText(show_frame, "No face detected", (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return
    
    # 왼쪽 맨 앞의 얼굴을 운전자로 가정
    faces = sorted(faces, key=lambda x: (x[0], -x[1]))
    driver_face = faces[0]
    
    for i, (x, y, w, h) in enumerate(faces):
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        points = np.matrix([[p.x, p.y] for p in predictor(image, rect).parts()])

        if i == 0: # 운전자
            cv2.putText(show_frame, "Driver", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 졸음 감지 로직 적용
            show_parts = points[EYES]
            right_eye_EAR = getEAR(points[RIGHT_EYE])
            left_eye_EAR = getEAR(points[LEFT_EYE])
            mean_eye_EAR = (right_eye_EAR + left_eye_EAR) / 2 

            if not calibrated:
                # 캘리브레이션 단계
                total_EAR += mean_eye_EAR
                calibration_counter += 1
                if calibration_counter >= calibration_frames:
                    avg_EAR = total_EAR / calibration_counter
                    min_EAR = avg_EAR * 0.85  # 평균 EAR의 70%를 임계값으로 설정
                    calibrated = True
                    print(f"Calibration completed. min_EAR set to: {min_EAR:.3f}")
                color = (255, 255, 0)
                status = 'Calibrating'
            else:
                if mean_eye_EAR > min_EAR:
                    color = (0, 255, 0)
                    status = 'Awake'
                    number_closed -= 1
                    if number_closed < 0:
                        number_closed = 0
                else:
                    color = (0, 0, 255)
                    status = 'Sleep'
                    number_closed += 1
                            
                sign = 'Sleep count : ' + str(number_closed) + ' / ' + str(closed_limit)

            # 하품 인식
            mouth_points = points[MOUTH]
            mouth_MAR = getMAR(mouth_points)
            min_MAR = 0.6  # 하품으로 간주할 최소 MAR 값
            
            # 하품 인식 조건: 입이 벌어지고 눈이 감긴 상태
            if mouth_MAR > min_MAR and mean_eye_EAR < min_EAR:
                yawn_count += 1
                if yawn_count >= yawn_limit:
                    current_time = time.time()
                    if current_time - last_alarm_time > alarm_interval:
                        print("Yawning detected")
                        winsound.PlaySound("./alarm.wav", winsound.SND_FILENAME | winsound.SND_ASYNC)
                        last_alarm_time = current_time
                    yawn_count = 0  # 하품 횟수 초기화

            # 졸음 확정시 알람 설정
            if number_closed > closed_limit:
                current_time = time.time()
                if current_time - last_alarm_time > alarm_interval:
                    print("Alarm condition met")
                    winsound.PlaySound("./alarm.wav", winsound.SND_FILENAME | winsound.SND_ASYNC)
                    last_alarm_time = current_time
                number_closed = 0  # 눈 감김 횟수 초기화

        else: # 동승자
            cv2.putText(show_frame, "Passenger", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


    # 화면에 상태 및 하품 횟수 표시
    cv2.putText(show_frame, status, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
    if calibrated:
        cv2.putText(show_frame, sign, (10, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(show_frame, 'Yawn count : ' + str(yawn_count) + ' / ' + str(yawn_limit), (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# 인수 파싱 함수
def arg_parse():
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file", default="yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile", default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed", default="416", type=str)

    return parser.parse_args()

# 이미지 전처리 함수
def prep_image(img, inp_dim):
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
classes = load_classes("C:/Users/82103/Desktop/codeit/summerproject/test3(동승자 추가)/coco.names")  # YOLO 클래스 이름 파일 경로(상대 경로)

# 신경망 설정
print("Loading.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Success")

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

# 메인 루프
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

    # YOLO 모델을 사용하여 객체 탐지
    with torch.no_grad():
        output = model(Variable(img), CUDA)
    output = write_results(output, confidence, num_classes, nms_conf=nms_thesh)

    if isinstance(output, int):
        show_frame = orig_im.copy()
        detect_drowsiness(orig_im)
        cv2.imshow(title_name, show_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue  # 객체가 탐지되지 않은 경우 다음 프레임으로
    
    im_dim = im_dim.repeat(output.size(0), 1)
    scaling_factor = torch.min(inp_dim / im_dim, 1)[0].view(-1, 1)

    output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
    output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2
    output[:, 1:5] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
        output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

    # 스마트폰 탐지 시 알람 설정(운전자 근처에 있을 때만)
    for obj in output:
        cls = int (obj[-1])
        if classes[cls] in ['remote', 'cell phone']:
            x1, y1, x2, y2 = obj[1:5]
            if driver_face is not None:
                dx, dy, dw, dh = driver_face
                # 스마트폰이 운전자 얼굴 근처에 있는지 확인
                if (x1 > dx and x1 < dx+dw) or (x2 > dx and x2 < dx+dw):
                    current_time = time.time()
                    if current_time - last_alarm_time > alarm_interval:
                        print("smart phone detected near driver! plz 집.중.")
                        winsound.PlaySound("./alarm.wav", winsound.SND_FILENAME | winsound.SND_ASYNC)
                        last_alarm_time = current_time

    list(map(lambda x: write(x, orig_im), output))

    show_frame = orig_im.copy()
    detect_drowsiness(orig_im)

    # 화면에 결과 표시
    cv2.imshow(title_name, show_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
torch.cuda.empty_cache()
