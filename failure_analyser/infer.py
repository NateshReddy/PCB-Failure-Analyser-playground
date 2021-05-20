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
import matplotlib.pyplot as plt
import matplotlib.image as mping
from utils import *
import os 


if os.path.exists('testing'):
    shutil.rmtree('testing')
os.mkdir("testing")
os.mkdir("testing/difference_image")
os.mkdir("testing/extracted_errors")

learn = load_learner('inception_2')

#extract defects
temp_file_path = '6_hold_out/00041000_temp.jpg'
test_file_path = '6_hold_out/00041000_test.jpg'
write_path_b = "testing/difference_image/diff_image_1.png"
write_path_w = "testing/difference_image/diff_image_2.png"
subtract_images(temp_file_path, test_file_path, write_path_b, 1) #black defects
subtract_images(test_file_path, temp_file_path, write_path_w, 2) #white defects

all_differenced_images = os.listdir('testing/difference_image')
for index, filename in enumerate(all_differenced_images):
    os.mkdir("testing/extracted_errors/extracts_" + str(index))
    image_path = "testing/difference_image/" + filename
    this_write_path = "testing/extracted_errors/extracts_" + str(index) + "/"
    hsv_lower = [0,150,50], #for red colour
    hsv_upper = [10,255,255]
    extract_contours_from_image(image_path, this_write_path, hsv_lower, hsv_upper)

os.mkdir("testing/all_extracted_errors")
cnt=0
for dirpath, dirs, files in os.walk("testing/extracted_errors"):
  for index, filename in enumerate(files):
    if filename.endswith(".png"):
        cnt += 1
        file_paths = [dirpath + "/" + filename]
        target_directory = "testing/all_extracted_errors/extract_" + str(cnt) + ".png"
        for file in file_paths:
          shutil.move(file, target_directory)
  
pred = []
for dirpath, dirs, files in os.walk("testing/all_extracted_errors"):
  for index, filename in enumerate(files):
    if filename.endswith(".png"):
        cnt += 1
        file_paths = [dirpath + "/" + filename]
        print(file_paths[0])
        img = open_image(file_paths[0])
        prediction = learn.predict(img)
        p = learn.data.classes[prediction[1].item()]
        pred.append(p)

print('predictions:', pred)