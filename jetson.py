# importing libraries
import cv2
import numpy as np
import torch
import kornia as K
import yaml

# Global constant
color_distance_threshold = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
skip_frames = 3 # only process 1 frame per this number of skip frame to save resource
learning_rate_angle = 0.2 # Learning rate for angle feedback
learning_rate_speed = 0.4 # Learning rate for speed feedback
min_speed = 0.1 # minimum robot speed as fraction of highest speed
pixel_to_mm = 3.285521454
robot_height = 215
robot_wide = 300
# Line color
line_color = (255, 0, 0)
direction_line_color = (100, 0, 255)
text_location1 = (50, 130)
text_location2 = (50, 160)
text_location3 = (50, 190)
text_location4 = (50, 230)
font = cv2.FONT_HERSHEY_SIMPLEX
err = 0.0001
# Camera matrix
yaml_file = "cam3.yaml"


def undistort(img):
    with open(yaml_file, "r") as f:
        data = yaml.load(f, Loader = yaml.loader.SafeLoader)
        mtx=np.array(data['camera_matrix'])
        dist=np.array(data['dist_coeff'])
    h, w = img.shape[0], img.shape[1]
    # Refining the camera matrix using parameters obtained by calibration
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    # Method 1 to undistort the image
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # # Method 2 to undistort the image
    # mapx, mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
    # dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
    return dst


def find_ax_by_c(x1, y1, x2, y2):
    a = (y1 - y2) / (x1 - x2 + err)
    b = y1 - x1 * a
    return a, b


def find_mid_dis(a_mid, b_mid, y_lower, w, h):
    x = (w//2-b_mid)/(a_mid + err)
    if x < h and x > 0:
        return y_lower*(robot_wide/np.abs(h-x) + 1)
    elif x < 0:
        return y_lower*(robot_wide/(h+np.abs(x))+1)
    else:
        return y_lower*(robot_wide/np.abs(x-h) - 1)

def mid_line(left_line, right_line, w, h):
    ptl1, ptl2 = left_line[0], left_line[1]
    ptr1, ptr2 = right_line[0], right_line[1]

    # calculate the center line
    a_l, b_l = find_ax_by_c(ptl1[1], ptl1[0], ptl2[1], ptl2[0])
    a_r, b_r = find_ax_by_c(ptr1[1], ptr1[0], ptr2[1], ptr2[0])

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
        a_mid, b_mid = find_ax_by_c(x_mid, y_mid, x_inter, y_inter)
    else:
        x_mid, y_mid = (x_ml + x_mr2) / 2, (y_ml + y_mr2) / 2
        a_mid, b_mid = find_ax_by_c(x_mid, y_mid, x_inter, y_inter)

    x_mid_lower = h
    y_mid_lower = a_mid * x_mid_lower + b_mid

    x_mid_upper = 0
    y_mid_upper = b_mid

    return (int(y_mid_lower), int(h)), (int(y_mid_upper), int(x_mid_upper)), (a_mid, b_mid)


def lane_making(img):
    arr = np.array(img)
    arr[:, 0], arr[:, -1] = arr[:, -1], arr[:, 0]
    x_rgb = torch.tensor(arr, device=device).unsqueeze(0) / 255
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
    if len(contours) < 2:
        return None
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
        return None
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
cap = cv2.VideoCapture(0)
# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video file")
# Read until video is completed
w, h = int(cap.get(3)), int(cap.get(4))
output = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'MJPG'), 20, (w, h))
num_frame = 0
angle, speed_feedback = 0, 0
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    # Too much computing power for undistort, so use only on PC
    # frame = undistort(frame)
    if num_frame % skip_frames == 0:
        # Just to prevent overflow
        num_frame = 0
        _lanes = lane_making(frame)
        if _lanes is not None:
            _left_line, _right_line = _lanes
            _mid_line = mid_line(_left_line, _right_line, w, h)
            # distance midline & mid of image
            a_mid, b_mid = _mid_line[2]
            y_lower = w//2 - _mid_line[0][0]
            dis_mid = find_mid_dis(a_mid, b_mid, y_lower, w, h)
            camera_axis_to_mid = np.clip(w // 2 - _mid_line[0][0], -w // 2, w // 2)
            angle = (angle + learning_rate_angle * np.clip(a_mid, -np.pi/4, np.pi/4)) / (1 + learning_rate_angle)
            speed_feedback_raw = 1 - abs(2 * np.clip(_mid_line[1][0]/w, min_speed, 1-min_speed) - 1)
            speed_feedback = (speed_feedback + learning_rate_speed * speed_feedback_raw) / (1 + learning_rate_speed)
    num_frame += 1
    cv2.line(frame, _left_line[0], _left_line[1], line_color, 3)
    cv2.line(frame, _right_line[0], _right_line[1], line_color, 3)
    cv2.line(frame, _mid_line[0], _mid_line[1], direction_line_color, 3)
    cv2.putText(frame, 'speed_feedback: ' + str(round(speed_feedback, 3)), text_location1, font, 1, line_color, 1, cv2.LINE_AA)
    cv2.putText(frame, 'angle: ' + str(round(np.rad2deg(angle), 3)), text_location2, font, 1, line_color, 1, cv2.LINE_AA)
    cv2.putText(frame, 'camera_axis_to_mid: ' + str(camera_axis_to_mid), text_location3, font, 1, line_color, 1, cv2.LINE_AA)
    cv2.putText(frame, 'dis_mid: ' + str(round(dis_mid, 3)), text_location4, font, 1, line_color, 1, cv2.LINE_AA)
    cv2.circle(frame, (w//2, h), 5, direction_line_color, -1)
    output.write(frame)

# When everything done, release the video capture object & Closes all the frames
cap.release()
output.release()
