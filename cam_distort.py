"""Camera calibrate using multiple cams
This code is adapted from: https://github.com/EveryWhereLab/camera-calibration-using-opencv-python
and https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
@author: Le Thai Bach"""

from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import glob
import yaml
import os
h, w = (960, 1280) # image size in pixels

"""
This method is not completed
"""
def undistort_point(x, y, yaml_file):
    x = np.array(x)
    y = np.array(y)
    with open(yaml_file, "r") as f:
        data = yaml.load(f, Loader = yaml.loader.SafeLoader)
        print(data)
        mtx=np.array(data['camera_matrix'])
        
        dist=np.array(data['dist_coeff'])
    r = np.sqrt(x**2+y**2)


"""
This method unpack yaml file containing the calibration information and return
the mapping matrix of the distortion
"""
def unpack_yaml(yaml_file):
    with open(yaml_file, "r") as f:
        data = yaml.load(f, Loader = yaml.loader.SafeLoader)
        print(data)
        mtx=np.array(data['camera_matrix'])
        dist=np.array(data['dist_coeff'])
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    # Method 2 to undistort the image
    mapx, mapy = cv.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
    return mapx, mapy


"""
This method undistort image matrix from image as numpy array
"""
def undistort(img, mapx, mapy):       
    dst = cv.remap(img,mapx,mapy,cv.INTER_LINEAR)
    return dst

"""
This method undistort existing image file from the mapping matrix and output
result to predefine path
"""
def undistort_file(input_file, mapx, mapy, output_path = './'):
    img = cv.imread(input_file)
    dst = undistort(img, mapx, mapy)
    f = input_file.split('/')[-1]
    name = f.split('.')
    img_RGB = cv.imread(input_file)
    
    store_name = name[0] + "_undist." + name[1]
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plt.imshow(dst)
    plt.show()
    cv.imwrite(os.path.join(output_path, store_name), dst)