#================================================================
#
#   File name   : detection_demo.py
#   Author      : PyLessons
#   Created date: 2020-09-27
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : object detection image and video example
#
#================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.add_dll_directory('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\bin')
import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import detect_image, detect_realtime, detect_video, Load_Yolo_model, detect_video_realtime_mp
from yolov3.configs import *
from yolov3.yolov4 import read_class_names

image_path   = "./IMAGES/IMG_1497.jpg"
video_path   = "./IMAGES/1397918818234.mp4"

yolo = Load_Yolo_model()

pred_classes = [] # 예측된 Class들 저장하는 pred_classses
pred_classes = detect_image(yolo, image_path, "./IMAGES/IMG_1497_pred.jpg", input_size=YOLO_INPUT_SIZE, show=True, rectangle_colors=(255,0,0))

if len(pred_classes) == 0: print("검출 안 됐음. 다시 사진 업로드 해주세요.") # 어플로 이미지 입력 다시 받는 부분

CLASSES = YOLO_COCO_CLASSES # CLASSES
NUM_CLASS = read_class_names(CLASSES)  # CLASSES NUM

for i in pred_classes: # 예측한 클래스들 출력
    print(NUM_CLASS[i])


#detect_video(yolo, video_path, "./IMAGES/1397918818234_pred.mp4", input_size=YOLO_INPUT_SIZE, show=False, rectangle_colors=(255,0,0))
#detect_realtime(yolo, '', input_size=YOLO_INPUT_SIZE, show=True, rectangle_colors=(255, 0, 0))

#detect_video_realtime_mp(video_path, "Output.mp4", input_size=YOLO_INPUT_SIZE, show=False, rectangle_colors=(255,0,0), realtime=False)
