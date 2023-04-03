# importing libraries
import cv2
import numpy as np
import math
import torch
import kornia as K
# Global constant
color_distance_threshold = 8
black_threshold = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Line color
line_color = (100,0,255)
direction_line_color = (255,100,0)
text_location = (50,100)
err = 0.0001
def detect_mid_line(ptl1, ptl2, ptr1, ptr2):
    y_l1, x_l1 = ptl1
    y_l2, x_l2 = ptl2
    y_r1, x_r1 = ptr1
    y_r2, x_r2 = ptr2

    #calculate the center line
    a_l, a_r = (y_l1-y_l2)/(x_l1-x_l2+err), (y_r1-y_r2)/(x_r1-x_r2+err)
    b_l, b_r = y_l1 - x_l1 * a_l, y_r1 - x_r1 * a_r

    #get intersect point
    x_inter = (b_r-b_l)/(a_l-a_r+err)
    y_inter = x_inter*a_l + b_l

    #get mid_point left
    x_ml = x_inter + 1/err
    y_ml = x_ml * a_l + b_l
    d_2 = np.square(y_inter - y_ml) + np.square(x_inter - x_ml)

    #get mid_point right
    #delta = b^2 - 4ac
    a_mr = np.square(a_r) + 1
    b_mr = -2*(x_inter+y_inter*a_r-a_r*b_r)
    c_mr = -(d_2 - np.square(x_inter) - np.square(y_inter) + 2*y_inter*b_r - np.square(b_r))
    delta = np.square(b_mr) - 4*a_mr*c_mr

    #get x_mid_right and y_mid_right
    x_mr1 = (-b_mr - np.sqrt(delta)) / (2*a_mr+err)
    x_mr2 = (-b_mr + np.sqrt(delta)) / (2*a_mr+err)

    y_mr1 = a_r * x_mr1 + b_r
    y_mr2 = a_r * x_mr2 + b_r

    #return mid_line
    if x_mr1 > x_inter:
        x_mid, y_mid = (x_ml + x_mr1)/2, (y_ml + y_mr1)/2
    else:
        x_mid, y_mid = (x_ml + x_mr2)/2, (y_ml + y_mr2)/2
    return int(y_inter), int(x_inter), int(y_mid), int(x_mid)
def lane_making(img):
    arr = np.array(img)
    arr[:, 0], arr[:, -1] = arr[:, -1], arr[:, 0]
    x_rgb = torch.tensor(arr, device=device).unsqueeze(0) / 255
    x_rgb = torch.transpose(x_rgb, dim0=0, dim1=-1)[..., 0]
    x_rgb = x_rgb.unsqueeze(0)
    # Convert to Lab
    img_blurred = K.color.rgb_to_lab(x_rgb)
    # Seperate the color components and lightning
    lightning = img_blurred[0, 0, :, :]
    mean_lightning = torch.mean(lightning)
    color = img_blurred[0, 1:, :, :]
    color_dist = torch.sqrt(torch.sum(torch.pow(color, exponent=2), dim=0))
    # Apply adaptive thresholding for lightning matrix and global thresholding for color matrix
    # We must filter black pixel before feeding to otsu
    lightning = torch.where(lightning > black_threshold, lightning, mean_lightning)
    color_thresh = color_dist < color_distance_threshold
    lightning = lightning.to("cpu").numpy().astype(np.uint16)
    _, lightning_thresh = cv2.threshold(lightning, 0, 255, cv2.THRESH_OTSU)
    lightning_thresh = torch.tensor(lightning_thresh.astype(bool), device=device)
    mixture = torch.logical_and(color_thresh, lightning_thresh)
    thresh_result = torch.where(mixture, 1, 0)
    # Applying the Canny Edge filter
    edges = cv2.Canny(np.uint8(thresh_result.to("cpu").numpy() * 255), 0, 255)
    lines = cv2.HoughLines(edges, 1, np.pi / 90, 50, None, 0, 0)
    cv2.putText(img, 'None', text_location, cv2.FONT_HERSHEY_SIMPLEX, 1, line_color, 2)
    cv2.imshow('Frame', edges)
    return
# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('../../data/line_trace/bacho/congthanh_solution.mp4')
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