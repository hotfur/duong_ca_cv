"""Program to detect line
@author Vuong Kha Sieu
@author Nguyen Hoang An
@author Le Thai Bach
"""

import cv2
import numpy as np
import queue

# Global constant
color_distance_threshold = 8
black_threshold = 128
morphology_iterations = 3

if __name__ == '__main__':
    path = '../../data/line_trace/congthanh_solution/'
    img = cv2.imread(cv2.samples.findFile(path + "142" + ".png"))
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
    bw = cv2.medianBlur(morph, ksize=5)
    # Applying the Canny Edge filter
    edges = cv2.Canny(np.uint8(bw), 0, 255)
    lines = cv2.HoughLines(edges, 1, np.pi / 90, 50, None, 0, 0)
    searched_lines = set()
    h, w = bw.shape
    master_q = queue.PriorityQueue()
    for line1_indx in range(10):
        aux_q = queue.PriorityQueue()
        line1 = lines[line1_indx]
        searched_lines.add(line1_indx)
        for line2_indx in range(15):
            line2 = lines[line2_indx]
            if line2_indx not in searched_lines:
                black = np.zeros(bw.shape, dtype=np.int16)
                pt11 = (int(line1[0][0] / np.cos(line1[0][1])), 0)
                pt12 = (int((line1[0][0] - (h - 1) * np.sin(line1[0][1])) / np.cos(line1[0][1])), h - 1)
                pt21 = (int(line2[0][0] / np.cos(line2[0][1])), 0)
                pt22 = (int((line2[0][0] - (h - 1) * np.sin(line2[0][1])) / np.cos(line2[0][1])), h - 1)
                pts = np.array([pt11, pt12, pt21, pt22])
                cv2.fillPoly(black, [pts], color=255)
                area = np.mean(bw, where=black > black_threshold) / black_threshold
                if area > 1:
                    aux_q.put((2 - area, line1, line2))
        if not aux_q.empty():
            best_pair = aux_q.get()
            master_q.put(best_pair)
    searched_lines = []
    while not master_q.empty():
        priority, line1, line2 = master_q.get()
        searched_lines.append(line1)
        searched_lines.append(line2)
    for r_theta in searched_lines:
        arr = np.array(r_theta[0], dtype=np.float64)
        r, theta = arr
        # Stores the value of cos(theta) in a
        a = np.cos(theta)

        # Stores the value of sin(theta) in b
        b = np.sin(theta)

        # x0 stores the value rcos(theta)
        x0 = a * r

        # y0 stores the value rsin(theta)
        y0 = b * r

        # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
        x1 = int(x0 + 1000 * (-b))

        # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
        y1 = int(y0 + 1000 * (a))

        # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
        x2 = int(x0 - 1000 * (-b))

        # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
        y2 = int(y0 - 1000 * (a))

        # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
        # (0,0,255) denotes the colour of the line to be
        # drawn. In this case, it is red.
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite("test.jpg", img)


