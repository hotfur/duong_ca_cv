# importing libraries
import cv2
import kornia as K
import numpy as np
import torch

# Global constant
color_distance_threshold = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Line color
line_color = (100, 0, 255)
direction_line_color = (255, 100, 0)
text_location = (50, 100)
err = 0.0001


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
    if len(contours) == 0:
        return
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
    if len(contours) == 0:
        return
    # Take only the two best contours as line.
    contours.sort(reverse=True)
    contours = contours[:2]
    # Write contours to image
    for moment, i, cnt in contours:
        rows, cols = img.shape[:2]
        [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        lefty = int(np.clip((-x * vy / vx) + y, -10000, 10000))
        righty = int(np.clip(((cols - x) * vy / vx) + y, -10000, 10000))
        cv2.line(img, (cols - 1, righty), (0, lefty), line_color, 2)
        cv2.drawContours(img, [cnt], 0, line_color, 3)
    # cv2.drawContours(img, contours, -1, line_color, 3)
    return img


cap = cv2.VideoCapture('../../data/line_trace/bacho/WIN_20230401_16_15_08_Pro.mp4')
# For Jetson Nano
# cap = cv2.VideoCapture('congthanh_solution.mp4')
if not cap.isOpened():
    print("Error opening video file")
output = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'MJPG'), 10, (int(cap.get(3)), int(cap.get(4))))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    processed = lane_making(frame)
    output.write(processed)
# When everything done, release the video capture object
cap.release()
output.release()
