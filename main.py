"""Program to detect line
@author Vuong Kha Sieu
@author Nguyen Hoang An
@author Le Thai Bach
"""

import cv2
import numpy as np

# Global constant
color_distance_threshold = 10
black_threshold = 128
morphology_iterations = 3

if __name__ == '__main__':
    path = '../../data/line_trace/congthanh_solution/'
    img = cv2.imread(cv2.samples.findFile(path + "0" + ".png"))
    # Gaussian filter to slightly reduce noise
    img_blurred = cv2.GaussianBlur(img, ksize=(3,3), sigmaX=1, sigmaY=1)
    # Convert to Lab
    img_blurred = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2Lab)
    # Seperate the color components and lightning
    lightning = img_blurred[:, :, 0]
    mean_lightning = np.mean(lightning)
    color = img_blurred[:, :, 1:] - 127
    color_dist = np.sqrt(np.sum(color.__pow__(2), axis=2))
    # Apply adaptive thresholding for lightning matrix and global thresholding for color matrix
    # We must filter black pixel before feeding to otsu
    lightning = np.where(lightning>black_threshold, lightning, int(mean_lightning))
    _, lightning_thresh = cv2.threshold(lightning, 0, 255, cv2.THRESH_OTSU)
    color_thresh = color_dist < color_distance_threshold
    mixture = np.logical_and(color_thresh, lightning_thresh.astype(bool))
    thresh_result = np.where(mixture, 255, 0).astype(np.int16)
    # Apply morphology transformation to filter noises
    morph = cv2.dilate(thresh_result, kernel=(3,3), iterations=morphology_iterations)
    morph = cv2.erode(morph, kernel=(3,3), iterations=morphology_iterations)
    # Apply median filter to remove excessive noises
    result = cv2.medianBlur(morph, ksize=5)
    cv2.imwrite("test.jpg", result)


