from email.mime import image
import io
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import torchvision
import numpy as np
# load model


model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)


# PATH = "./model.pth"

device = 'cpu'

# model.load_state_dict(torch.load('model.pth'))
model = model.to(device)
# model = torch.load('./model.pth', map_location='cpu')


model.eval()


def apply_nms(orig_prediction, iou_thresh=0.3):
    keep = torchvision.ops.nms(
        orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    return final_prediction
# image -> tensor


def transform_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1))

    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    image = torch.FloatTensor(image)

    return image

# predict


def get_prediction(image):
    # image = cv2.imread('sample_image.jpg')

    image = image.to(device)

    detections = model(image)[0]
    apply_nms(detections, iou_thresh=0.3)
    return detections
