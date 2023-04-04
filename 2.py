# importing libraries
import cv2
import numpy as np
import torch
import kornia as K
# Global constant
color_distance_threshold = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Line color
line_color = (255,100,0)
direction_line_color = (100,0,255)
text_location = (50,100)
err = 0.0001


def mid_line(left_line, right_line, err=0.0001):
    ptl1, ptl2 = left_line[0], left_line[1]
    ptr1, ptr2 = right_line[0], right_line[1]
    y_l1, x_l1 = ptl1
    y_l2, x_l2 = ptl2
    y_r1, x_r1 = ptr1
    y_r2, x_r2 = ptr2

    # calculate the center line
    a_l, a_r = (y_l1 - y_l2) / (x_l1 - x_l2 + err), (y_r1 - y_r2) / (x_r1 - x_r2 + err)
    b_l, b_r = y_l1 - x_l1 * a_l, y_r1 - x_r1 * a_r

    # get intersect point
    x_inter = (b_r - b_l) / (a_l - a_r + err)
    y_inter = x_inter * a_l + b_l

    # get mid_point left
    x_ml = x_inter + 1 / err
    y_ml = x_ml * a_l + b_l
    d_2 = np.square(y_inter - y_ml) + np.square(x_inter - x_ml)

    # get mid_point right
    # delta = b^2 - 4ac
    a_mr = np.square(a_r) + 1
    b_mr = -2 * (x_inter + y_inter * a_r - a_r * b_r)
    c_mr = -(d_2 - np.square(x_inter) - np.square(y_inter) + 2 * y_inter * b_r - np.square(b_r))
    delta = np.square(b_mr) - 4 * a_mr * c_mr

    # get x_mid_right and y_mid_right
    x_mr1 = (-b_mr - np.sqrt(delta)) / (2 * a_mr + err)
    x_mr2 = (-b_mr + np.sqrt(delta)) / (2 * a_mr + err)

    y_mr1 = a_r * x_mr1 + b_r
    y_mr2 = a_r * x_mr2 + b_r

    # return mid_line
    if x_mr1 > x_inter:
        x_mid, y_mid = (x_ml + x_mr1) / 2, (y_ml + y_mr1) / 2
    else:
        x_mid, y_mid = (x_ml + x_mr2) / 2, (y_ml + y_mr2) / 2

    return ((int(y_inter), int(x_inter)), (int(y_mid), int(x_mid)))


def lane_making(img):
    arr = np.array(img)
    arr[:, 0], arr[:, -1] = arr[:, -1], arr[:, 0]
    x_rgb = torch.tensor(arr, dtype=torch.half, device=device).unsqueeze(0) / 255
    x_rgb = torch.transpose(x_rgb, dim0=0, dim1=-1)[..., 0]
    x_rgb = x_rgb.unsqueeze(0)
    # Convert to Lab colorspace
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
    # Return black and white image to CPU for contour finding
    bw = torch.squeeze(thresh_result).cpu().numpy().astype(np.uint8) * 255
    contours = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    contours = contours[0] if len(contours) == 2 else contours[1]
    # Defensive programming
    if len(contours) == 0:
        return None, None
    # Filter contours by area and number of vertices
    contours_and_weights = []
    i = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 0:
            contours_and_weights.append((area, i, cnt))
        i += 1
    contours_and_weights.sort(reverse=True)
    # Filter contours by area powered by image momentum
    contours = []
    for cnt in range(min(6, len(contours_and_weights))):
        contour = contours_and_weights[cnt][2]
        moment = np.power(contours_and_weights[cnt][0], cv2.HuMoments(cv2.moments(contour))[0])
        contours.append((moment, cnt, contour))
    # Defensive programming
    if len(contours) < 2:
        return None, None
    lines = []
    # Take only the two best contours as line.
    contours.sort(reverse=True)
    contours = contours[:2]
    # Write contours to image
    for moment, i, cnt in contours:
        rows, cols = img.shape[:2]
        [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        lefty = int(np.clip((-x * vy / vx) + y, -1/err, 1/err))
        righty = int(np.clip(((cols - x) * vy / vx) + y, -1/err, 1/err))
        lines.append((cols - 1, righty))
        lines.append((0, lefty))
    return (lines[0], lines[1]), (lines[2], lines[3])


# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('../../data/line_trace/bacho/congthanh_solution.mp4')
if not cap.isOpened():
    print("Error opening video file")
output = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'MJPG'), 20, (int(cap.get(3)), int(cap.get(4))))
num_frame = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if num_frame % 4 == 0:
        # Just to prevent overflow
        num_frame = 0
        _left_line, _right_line = lane_making(frame)
        if _left_line is None:
            output.write(frame)
            continue
        _mid_line = mid_line(_left_line, _right_line)
    num_frame += 1

    cv2.line(frame, _left_line[0], _left_line[1], line_color, 2)
    cv2.line(frame, _right_line[0], _right_line[1], line_color, 2)
    cv2.line(frame, _mid_line[0], _mid_line[1], direction_line_color, 2)
    output.write(frame)

# When everything done, release the video capture object & Closes all the frames
cap.release()
output.release()