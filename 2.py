# importing libraries
import cv2
import math
import numpy as np
import torch
import kornia as K
# Global constant
color_distance_threshold = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Line color
line_color = (100,0,255)
direction_line_color = (255,100,0)
text_location = (50,100)
err = 0.0001

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
    bw = torch.squeeze(thresh_result).cpu().numpy().astype(np.uint8)*255
    contours = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    contours_and_weights = []
    i = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 0:
            contours_and_weights.append((area/len(cnt), i, cnt))
        i+=1
    contours_and_weights.sort(reverse=True)
    contours = []
    for cnt in range(min(2, len(contours_and_weights))):
        contours.append(contours_and_weights[cnt][2])
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    for cnt in contours:
        rows, cols = img.shape[:2]
        [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        lefty = int((-x * vy / vx) + y)
        righty = int(((cols - x) * vy / vx) + y)
        cv2.line(img, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
    # cv2.imshow('edges', img)
    return img
# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('../../data/line_trace/bacho/congthanh_solution.mp4')
# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video file")
# Read until video is completed
output = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'MJPG'), 10, (int(cap.get(3)), int(cap.get(4))))
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    processed=lane_making(frame)
    output.write(processed)
# When everything done, release the video capture object & Closes all the frames
cap.release()
cv2.destroyAllWindows()