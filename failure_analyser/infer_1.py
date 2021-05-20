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
from utils1 import *
import os 
import base64


def get_predictions(temp_b64, test_b64, unique_path):
    if os.path.exists("testing/" + unique_path):  # replace testing with unique path
        shutil.rmtree("testing/" + unique_path)
    os.mkdir("testing/" + unique_path)
    os.mkdir("testing/" + unique_path + "/difference_image")
    os.mkdir("testing/" + unique_path + "/extracted_errors")

    learn = load_learner('inception_2')

    #extract defects
    #temp_file_path = '6_hold_out/00041000_temp.jpg'
    #test_file_path = '6_hold_out/00041000_test.jpg'
    write_path_b = "testing/" + unique_path + "/difference_image/diff_image_1.png"
    write_path_w = "testing/" + unique_path + "/difference_image/diff_image_2.png"
    subtract_images(temp_b64, test_b64, write_path_b, 1) #black defects
    subtract_images(test_b64, temp_b64, write_path_w, 2) #white defects

    all_differenced_images = os.listdir("testing/" + unique_path + '/difference_image')
    for index, filename in enumerate(all_differenced_images):
        os.mkdir("testing/" + unique_path + "/extracted_errors/extracts_" + str(index))
        image_path = "testing/" + unique_path + "/difference_image/" + filename
        this_write_path = "testing/" + unique_path + "/extracted_errors/extracts_" + str(index) + "/"
        hsv_lower = [0,150,50], #for red colour
        hsv_upper = [10,255,255]
        extract_contours_from_image(image_path, this_write_path, hsv_lower, hsv_upper)

    os.mkdir("testing/" + unique_path + "/all_extracted_errors")
    cnt=0
    for dirpath, dirs, files in os.walk("testing/" + unique_path + "/extracted_errors"):
        for index, filename in enumerate(files):
            if filename.endswith(".png"):
                cnt += 1
                file_paths = [dirpath + "/" + filename]
                target_directory = "testing/" + unique_path + "/all_extracted_errors/extract_" + str(cnt) + ".png"
                for file in file_paths:
                    shutil.move(file, target_directory)
    
    pred = []
    folder_path = "testing/" + unique_path + "/all_extracted_errors"
    #curr_dir = os.getcwd()
    #f = os.path.join(curr_dir, folder_path)
    f = folder_path
    for dirpath, dirs, files in os.walk(folder_path):
        dic = {'folder': f, 'images': [], 'key': unique_path}
        for index, filename in enumerate(files):
            if filename.endswith(".png"):
                cnt += 1
                file_paths = [dirpath + "/" + filename]
                #print(file_paths[0])
                img = open_image(file_paths[0])
                prediction = learn.predict(img)
                p = learn.data.classes[prediction[1].item()]
                #pred.append(p)
                d1 = {'image': filename, 'prediction': p}
                dic['images'].append(d1)

    #print('predictions:', pred)
    return dic