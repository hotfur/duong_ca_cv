"""Camera calibrate using multiple cams
This code is adapted from: https://github.com/EveryWhereLab/camera-calibration-using-opencv-python
@author: Vuong Kha Sieu"""
from pathlib import Path
import numpy as np
import cv2 as cv
import yaml
from concurrent.futures import ThreadPoolExecutor

sensor_size = (3.58, 2.02)
def calibrate(input):
    CHESSBOARD_CORNER_NUM_X = 9
    CHESSBOARD_CORNER_NUM_Y = 6

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
    objp = np.zeros((CHESSBOARD_CORNER_NUM_X*CHESSBOARD_CORNER_NUM_Y,3), np.float32)
    objp[:,:2] = np.mgrid[0:CHESSBOARD_CORNER_NUM_X,0:CHESSBOARD_CORNER_NUM_Y].T.reshape(-1,2)

    images = input.glob("*")
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    for fname in images:
        img = cv.imread(str(fname))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (CHESSBOARD_CORNER_NUM_X,CHESSBOARD_CORNER_NUM_Y), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv.drawChessboardCorners(img, (CHESSBOARD_CORNER_NUM_X,CHESSBOARD_CORNER_NUM_Y), corners2, ret)

    img_dim = img.shape
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error

    converted_mtx = convert_cam_mat(np.asarray(mtx), img_dim)
    data = {'camera_matrix': converted_mtx.tolist(), 'dist_coeff': np.asarray(dist).tolist(), "error": mean_error/len(objpoints)}
    output = Path("output/").joinpath(input.name + ".yaml")
    with open(str(output), "w+") as f:
        yaml.dump(data, f)

def convert_cam_mat(mat, img_dim):
    """Convert the camera matrix from pixels to millimeter"""
    mat[0] *= sensor_size[0]/img_dim[1]
    mat[1] *= sensor_size[1]/img_dim[0]

    return mat


if __name__ == "__main__":
    # root directory of repo for relative path specification.
    root = Path(__file__).parent.parent.absolute()
    images_path = root.joinpath("data/")
    images = images_path.glob('*')
    with ThreadPoolExecutor() as executor:
        for i in images:
            executor.submit(calibrate, i)
