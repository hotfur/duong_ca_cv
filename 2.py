# importing libraries
import cv2
import math
import numpy as np
import torch
import kornia as K
import queue
from concurrent.futures import ThreadPoolExecutor
# Global constant
color_distance_threshold = 8
num_lines = 12 # CPU consumption
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Line color
line_color = (100,0,255)
direction_line_color = (255,100,0)
text_location = (50,100)
err = 0.0001
def find_best_lines(master_q,lines, line1_indx, shape, bw, all_highlighted_area):
    h, w = shape
    aux_q = queue.PriorityQueue()
    line1 = lines[line1_indx]
    for line2_indx in range(min(num_lines, len(lines))):
        line2 = lines[line2_indx]
        line_area = np.zeros((h, w), dtype=np.int16)
        pt11 = (int(line1[0][0] / np.cos(line1[0][1])), 0)
        pt12 = (int((line1[0][0] - (h - 1) * np.sin(line1[0][1])) / np.cos(line1[0][1])), h - 1)
        pt21 = (int(line2[0][0] / np.cos(line2[0][1])), 0)
        pt22 = (int((line2[0][0] - (h - 1) * np.sin(line2[0][1])) / np.cos(line2[0][1])), h - 1)
        pts = np.array([pt11, pt12, pt21, pt22])
        cv2.fillPoly(line_area, [pts], color=1)
        line_area = torch.tensor(line_area, device=device).bool()
        true_area = torch.sum(torch.logical_and(line_area, bw))
        good = true_area / torch.sum(line_area)
        significant = true_area / all_highlighted_area
        if good > 0.5 and significant > 0.05:
            try:
                aux_q.put((1 - significant, line1, line2))
            except:
                continue
    if not aux_q.empty():
        try:
            best_pair = aux_q.get()
            master_q.put(best_pair)
        except:
            return

def lane_making(img):
    arr = np.array(img)
    arr[:,0], arr[:,-1] = arr[:,-1], arr[:,0]
    x_rgb = torch.tensor(arr, device=device).unsqueeze(0) / 255
    x_rgb = torch.transpose(x_rgb, dim0=0, dim1=-1)[...,0]
    x_rgb = x_rgb.unsqueeze(0)
    # Convert to Lab
    img_blurred = K.color.rgb_to_lab(x_rgb)
    # Seperate the color components and lightning
    lightning = img_blurred[0, 0, :, :]
    std_mean_lightning = torch.std_mean(lightning)
    color = img_blurred[0, 1:, :, :]
    color_dist = torch.sqrt(torch.sum(torch.pow(color, exponent=2), dim=0))
    # Apply adaptive thresholding for lightning matrix and global thresholding for color matrix
    lightning = lightning > (std_mean_lightning[0] + std_mean_lightning[1])
    color_thresh = color_dist < color_distance_threshold
    mixture = torch.logical_and(color_thresh, lightning)
    thresh_result = torch.where(mixture, 1.0, 0.0)
    # Applying the Canny Edge filter
    _, edges = K.filters.canny(thresh_result[None,None,...], 0.01, 0.99)
    edges = torch.squeeze(edges).cpu().numpy().astype(np.uint8)*255
    bw = thresh_result.bool()
    all_highlighted_area = torch.sum(bw)
    lines = cv2.HoughLines(edges, 1, np.pi / 60, 50, None, 0, 0)
    if lines is None: return
    master_q = queue.PriorityQueue()
    with ThreadPoolExecutor() as e:
        for line1_indx in range(min(num_lines, len(lines))):
            e.submit(find_best_lines,master_q,lines, line1_indx, bw.shape, bw, all_highlighted_area)
    searched_lines = []
    while not master_q.empty():
        try:
            priority, line1, line2 = master_q.get()
        except:
            break
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
    #cv2.imshow('edges', img)
    return img
# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('../../data/line_trace/bacho/WIN_20230401_16_16_01_Pro.mp4')
# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video file")
# Read until video is completed
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    lane_making(frame)
# When everything done, release the video capture object & Closes all the frames
cap.release()
cv2.destroyAllWindows()