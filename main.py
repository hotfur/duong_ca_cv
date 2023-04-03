"""Program to detect line
@author Vuong Kha Sieu
@author Nguyen Hoang An
@author Le Thai Bach
"""

import cv2
import numpy as np
import queue
import torch
import kornia as K

# Global constant
color_distance_threshold = 8
black_threshold = 50
morphology_iterations = 3

if __name__ == '__main__':
    path = '../../data/line_trace/congthanh_solution/'
    img = cv2.imread(cv2.samples.findFile(path + "0" + ".png"))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x_bgr: torch.tensor = K.image_to_tensor(img)
    x_bgr = x_bgr.unsqueeze(0)
    x_bgr = x_bgr.to(device)
    x_rgb: torch.tensor = K.color.bgr_to_rgb(x_bgr) / 255
    # Gaussian filter to slightly reduce noise
    gauss = K.filters.GaussianBlur2d(kernel_size=(3,3), sigma=(1.0,1.0))
    img_blurred = gauss(x_rgb)
    # Convert to Lab
    img_blurred = K.color.rgb_to_lab(img_blurred)
    # Seperate the color components and lightning
    img_blurred = img_blurred.squeeze(0)
    lightning = img_blurred[0, :, :]
    mean_lightning = torch.mean(lightning)
    color = img_blurred[1:, :, :]
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
    # Apply morphology transformation to filter noises
    kernel = torch.tensor([[0, 1, 0],[1, 1, 1],[0, 1, 0]], dtype=torch.float32, device=device)
    morph = K.morphology.dilation(thresh_result[None,None,...], kernel=kernel)
    morph = K.morphology.dilation(morph, kernel=kernel)
    # Apply median filter to remove excessive noises
    bw = K.filters.median_blur(morph, kernel_size=5)
    # Applying the Canny Edge filter
    _, edges = K.filters.canny(bw, 0.01, 0.99)
    edges = edges.squeeze()*255
    edges = edges.to("cpu").numpy().astype(np.uint8)
    all_highlighted_area = torch.sum(bw)
    bw = bw.bool()
    lines = cv2.HoughLines(edges, 1, np.pi / 90, 50, None, 0, 0)
    searched_lines = set()
    _, _, h, w = bw.shape
    master_q = queue.PriorityQueue()
    for line1_indx in range(10):
        aux_q = queue.PriorityQueue()
        line1 = lines[line1_indx]
        searched_lines.add(line1_indx)
        for line2_indx in range(15):
            line2 = lines[line2_indx]
            if line2_indx not in searched_lines:
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
                    try: aux_q.put((1 - significant, line1, line2))
                    except: continue
        if not aux_q.empty():
            try:
                best_pair = aux_q.get()
                master_q.put(best_pair)
            except:
                continue
    searched_lines = []
    #for i in range(6):
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


