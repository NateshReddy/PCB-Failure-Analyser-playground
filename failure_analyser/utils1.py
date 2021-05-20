import shutil
import cv2
import numpy as np
import json
import glob
import pandas as pd
import zipfile
import matplotlib.pyplot as plt
#from pylab import rcParams
from sklearn.utils import resample
from fastai.vision import *
import torch, torchvision
from fastai.metrics import error_rate # 1 - accuracy
from fastai.callbacks import EarlyStoppingCallback,SaveModelCallback
import base64

def my_loss(preds,target):
  if isinstance(preds, tuple):
    loss = sum((F.cross_entropy(o,target) for o in preds))
  else:
    loss = F.cross_entropy(preds,target)
  return loss

# to extract differenced images
def subtract_images(image_1_b64, image_2_b64, write_path, choice):
    #image1 = cv2.imread(image_path_1)
    #image2 = cv2.imread(image_path_2)

    im_bytes1 = base64.b64decode(image_1_b64)
    im_arr1 = np.frombuffer(im_bytes1, dtype=np.uint8)  # im_arr is one-dim Numpy array
    image1 = cv2.imdecode(im_arr1, flags=cv2.IMREAD_COLOR)

    im_bytes2 = base64.b64decode(image_2_b64)
    im_arr2 = np.frombuffer(im_bytes2, dtype=np.uint8)  # im_arr is one-dim Numpy array
    image2 = cv2.imdecode(im_arr2, flags=cv2.IMREAD_COLOR)

    difference = cv2.subtract(image1, image2)
    Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    difference[mask != 255] = [0, 0, 255]
    image1[mask != 255] = [0, 0, 255]
    image2[mask != 255] = [0, 0, 255]
    if choice == 1: #to extract black over white errors
      cv2.imwrite(write_path, image1)
    elif choice == 2:#to extract white over black errors
      cv2.imwrite(write_path, image2)

# to extract errors from each image
def extract_contours_from_image(image_path, write_path, hsv_lower, hsv_upper):
    image = cv2.imread(image_path)
    original = image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_lower = np.array(hsv_lower)
    hsv_upper = np.array(hsv_upper)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper) #if red pixel, then 255 otherwise 0
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
    cnts, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    offset = 20
    ROI_number = 0
    if cnts!=None:
      for c in cnts:
          x, y, w, h = cv2.boundingRect(c)
          cv2.rectangle(image, (x - offset, y - offset), (x + w + offset, y + h + offset), (36, 255, 12), 2)
          ROI = original[y - offset:y + h + offset, x - offset:x + w + offset]
          try:
              cv2.imwrite(write_path + 'contour_{}.png'.format(ROI_number), ROI)
          except:
              print("skipping image " + image_path + "for counter" + str(ROI_number))
          ROI_number += 1
    else:
      print("No counters found at " + image_path)
