from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import glob
import yaml
import os
h, w = (960, 1280) # image size in pixels

def undistort_point(x, y, yaml_file):
    x = np.array(x)
    y = np.array(y)
    with open(yaml_file, "r") as f:
        data = yaml.load(f, Loader = yaml.loader.SafeLoader)
        print(data)
        mtx=np.array(data['camera_matrix'])
        
        dist=np.array(data['dist_coeff'])
    r = np.sqrt(x**2+y**2)
    
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

def undistort(img, mapx, mapy):       
    dst = cv.remap(img,mapx,mapy,cv.INTER_LINEAR)
    return dst
    
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