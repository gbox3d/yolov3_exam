# ================================================================
#
#   File name   : detection_demo.py
#   Author      : PyLessons
#   Created date: 2020-05-08
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : object detection image and video example
#
# ================================================================
# %%
import os
import sys
sys.path.append("../")

from yolov3.configs import *
from yolov3.utils import load_yolo_weights, detect_image, detect_video, detect_realtime,image_preprocess,postprocess_boxes,nms,draw_bbox
from yolov3.yolov3 import Create_Yolov3
import tensorflow as tf
import numpy as np
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import PIL.Image as Image
import matplotlib.pyplot as plt

from IPython.display import display

print("tf 버전: ", tf.__version__)
print("즉시 실행 모드: ", tf.executing_eagerly())
print("GPU ", "사용 가능" if tf.config.experimental.list_physical_devices("GPU") else "사용 불가능")

print(f'cv version {cv2.__version__}')

# tf.debugging.set_log_device_placement(True)

# %%
input_size = YOLO_INPUT_SIZE
Darknet_weights = YOLO_DARKNET_WEIGHTS
if TRAIN_YOLO_TINY:
    Darknet_weights = YOLO_DARKNET_TINY_WEIGHTS


# video_path = "./IMAGES/street_drive.mp4"

yolo = Create_Yolov3(input_size=input_size,CLASSES='../' + YOLO_COCO_CLASSES)
load_yolo_weights(yolo, '../'+Darknet_weights)  # use Darknet weights

print(f'weight data load ok {Darknet_weights}')

# %%
image_path = "../IMAGES/kite.jpg"

#이미지 로딩 & 전처리 
original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# 0~255, 0~1 사이 소수로 바꾸고 크기도 416 싸이즈 안에 맞춰 집어 넣는다. 
image_data = image_preprocess(np.copy(original_image), [
    input_size, input_size])

# plt.figure()
# plt.imshow(image_data)
# plt.colorbar()
# plt.grid(False)
# plt.show()

display( Image.fromarray((image_data*256).astype('uint8')))


# %%
# 검출박스 구하기
image_data = tf.expand_dims(image_data, 0)

YoloV3 = yolo

pred_bbox = YoloV3.predict(image_data)
pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
pred_bbox = tf.concat(pred_bbox, axis=0)

print(f'found box : {len(pred_bbox)}')

# %%



bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold=0.3)
print(f'after post-process found box : {len(bboxes)}')

bboxes = nms(bboxes, iou_threshold=0.45, method='nms')
print(f'after nms found box : {len(bboxes)}')

# bboxes
# %%

image = draw_bbox(original_image, bboxes, CLASSES='../'+YOLO_COCO_CLASSES, rectangle_colors='')


# %%
display(Image.fromarray(image))

# plt.figure()
# plt.imshow(image)
# plt.colorbar()
# plt.grid(False)
# plt.show()

