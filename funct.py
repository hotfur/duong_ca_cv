import Jetson.GPIO as GPIO
import time
import cv2
import numpy as np
import math
from operator import itemgetter
from numpy.linalg import solve
def bigway(frame,noise_thres):
    line_range = cv2.GaussianBlur(frame.copy(),(5,5),1)
    line_range[:int(len(frame) * 0.4), :] = 0
    line_range = cv2.inRange(line_range,noise_thres,(255,255,255)) 

    dst = cv2.Canny(line_range, 50, 250, None, 3)
    lines = cv2.HoughLines(dst, 1, np.pi / 180, 45, None, 0, )
    line_range_color = cv2.cvtColor(line_range, cv2.COLOR_GRAY2BGR)
    lines_function =[]

    #Draw middle line of frame
    # cv2.line(line_range_color, (int(line_range_color.shape[1]*2/5),0 ), (int(line_range_color.shape[1]*2/5), line_range_color.shape[0]), color= [255,255,255] ,thickness= 3)
    # cv2.line(line_range_color, (int(line_range_color.shape[1]*1/2),0 ), (int(line_range_color.shape[1]*1/2), line_range_color.shape[0]), color= [240,255,255] ,thickness= 3)
    # cv2.line(line_range_color, (int(line_range_color.shape[1]*3/5),0 ), (int(line_range_color.shape[1]*3/5), line_range_color.shape[0]), color= [255,255,255] ,thickness= 3)
    
    if lines is not None:
        for line in lines:
            rho = line[0][0]
            theta = line[0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            # take out 2 point of lines
            #pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            #pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            if not(-0.1 < a/(b+0.001) < 0.1):
                lines_function.append([a,b,-rho])

    mid_thres = [frame.shape[1] / 2, frame.shape[0] - 2]
    left_lines = [i for i in lines_function if (-i[1] * frame.shape[0] - i[2]) / (a + 0.01) > frame.shape[1] / 2]
    if len(left_lines) != 0:
        left_lines.sort(key=itemgetter(0))
        left_function = left_lines[int(len(left_lines)/2)]
        #pt1 = (-1000, int(-(-1000*left_function[0]+left_function[2])/left_function[1]))
        #pt2 = (1000, int(-(1000*left_function[0]+left_function[2])/left_function[1]))
        #cv2.line(line_range_color, pt1, pt2, (0,0,255), 4, cv2.LINE_AA)
    else:
        left_function = None
        intersection = None

    right_lines = [i for i in lines_function if (-i[1] * frame.shape[0] - i[2]) / (a + 0.01) < frame.shape[1] / 2]
    if len(right_lines) != 0:
        right_lines.sort(key=itemgetter(0))
        right_function = right_lines[int(len(right_lines)/2)]
        #pt1 = (-1000, int(-(-1000*right_function[0]+right_function[2])/right_function[1]))
        #pt2 = (1000, int(-(1000*right_function[0]+right_function[2])/right_function[1]))
        #cv2.line(line_range_color, pt1, pt2, (0,0,255), 4, cv2.LINE_AA)
        
    else: 
        right_function = None
        intersection = None
    if right_function != None and left_function != None:
        a = np.array([
            [left_function[0],left_function[1]],
            [right_function[0],right_function[1]]
        ])
        b = np.array([-left_function[2],-right_function[2]])
        intersection = solve(a,b)
        intersection = np.array(intersection,int)
        midPoint = findMidpoint(left_function,right_function)
        cv2.line(line_range_color, midPoint, intersection, (0,255,0), 4, cv2.LINE_AA)
    cv2.imshow("",line_range_color)
    return [left_function,right_function,intersection]
def findMidpoint(func1, func2):
    denominator1 = math.sqrt(func1[0]**2+func2[1]**2)
    denominator2 = math.sqrt(func2[0]**2+func2[1]**2)
    y = 1000
    a1 = func1[0] / denominator1
    b1 = func1[1] /denominator1
    c1 = func1[2] / denominator1

    a2 = func2[0] / denominator2
    b2 = func2[1] /denominator2
    c2 = func2[2] / denominator2

    x = ((b2-b1)*y+c2-c1) / (a1 - a2+0.001)
    return [int(x),y]
def stop_sign(input, min_red, mask):
    hsv = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
    lower_red = np.array([-10,70,80])
    upper_red = np.array([10,255,255])
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_red = mask * mask_red

    output = cv2.bitwise_and(input, input, mask = mask_red)

    #cv2.imshow('Red', output)
    #print('r:', sum(sum(np.array(mask_red))))

    if sum(sum(np.array(mask_red))) > min_red:
        return (True, sum(sum(np.array(mask_red))))
    else:
        return (False, 0)

def blue_sign(img, min_blue, mask):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90,90,50])
    upper_blue = np.array([130,255,255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    mask_blue = mask_blue*mask
    output = cv2.bitwise_and(img, img, mask = mask_blue)
    #print('b:', sum(sum(np.array(mask_blue))))

    #cv2.imshow('Blue', output)
    if sum(sum(np.array(mask_blue))) > min_blue:
        img_blur = cv2.medianBlur(img,3)
        img_gray = cv2.cvtColor(img_blur,cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(img_gray,cv2.HOUGH_GRADIENT, 1, 10, param1=200, param2=50,minRadius=10,maxRadius=0)

        if circles is None:
            return ('Blue', sum(sum(np.array(mask_blue))))
        else:
            if len(circles[0]) == 1:
                circles = np.uint16(np.around(circles))
                bot_left =  mask_blue[circles[0][0][1] :circles[0][0][1] + circles[0][0][2], circles[0][0][0] - int(circles[0][0][2]/2): circles[0][0][0]]
                bot_right = mask_blue[circles[0][0][1] :circles[0][0][1] + circles[0][0][2], circles[0][0][0]: circles[0][0][0] + int(circles[0][0][2]/2)]

                if sum(sum(bot_left)) < sum(sum(bot_right)):
                    return ('Left', sum(sum(np.array(mask_blue))))
                else:
                    return ('Right', sum(sum(np.array(mask_blue))))
            else:
                return ('Blue', sum(sum(np.array(mask_blue))))
    else:
        return ('False', 0)


def sign_detect(input, min_red, min_blue):
    mask = np.ones((len(input), len(input[0])), dtype='uint8')
    mask[int(len(input)/2), :] = 0
    
    # return structure: [stop, left, right]
    stop = stop_sign(input, min_red, mask)
    if stop[0] == True:
        return ([1, 0, 0], stop[1])
    else:
        blue = blue_sign(input, min_blue, mask)
        if blue[0] == 'Left':
            return ([0, 1, 0], blue[1])
        if blue[0] == 'Right':
            return ([0, 0, 1], blue[1])
        if blue[0] == 'Blue':
            return ([0, 0, 0], blue[1])
        else:
            return([0,0,0], 0)


def mid_line_equation(x_inter, y_inter, x_l, y_l, x_r, y_r):
    x_mid = (x_l + x_r)/2
    y_mid = (y_l + y_r)/2
    a_mid = (y_mid - y_inter)/(x_mid-x_inter+0.00001)
    b_mid = y_mid - a_mid * x_mid

    theta = np.arctan(-1/a_mid)
    rho = np.sin(theta) * b_mid

    print('Mid line equation:')

    print('y = {a}x + {b}'.format(a = a_mid, b = b_mid))
    print('theta = {th}, rho = {r}\n'.format(th = theta, r = rho))
    return (int(y_inter), int(x_inter), int(y_mid), int(x_mid), int(x_inter), int(y_inter), a_mid, b_mid)


def detect_mid_line(ptl1, ptl2, ptr1, ptr2):
    #calculate the center line
    y_l1 = ptl1[0]
    x_l1 = ptl1[1]
    y_l2 = ptl2[0]
    x_l2 = ptl2[1]

    a_l = (y_l1-y_l2)/(x_l1-x_l2+0.000001)
    b_l = y_l1 - x_l1 * a_l

    y_r1 = ptr1[0]
    x_r1 = ptr1[1]
    y_r2 = ptr2[0]
    x_r2 = ptr2[1]

    a_r = (y_r1-y_r2)/(x_r1-x_r2+0.0000001)
    b_r = y_r1 - x_r1 * a_r

    #get intersect point
    x_inter = (b_r-b_l)/(a_l-a_r+0.00001)
    y_inter = x_inter*a_l + b_l

    #get mid_point left
    x_ml = x_inter + 10000
    y_ml = x_ml * a_l + b_l
    d_2 = np.square(y_inter - y_ml) + np.square(x_inter - x_ml)

    #get mid_point right
    #delta = b^2 - 4ac
    a_mr = np.square(a_r) + 1
    b_mr = -2*(x_inter+y_inter*a_r-a_r*b_r)
    c_mr = -(d_2 - np.square(x_inter) - np.square(y_inter) + 2*y_inter*b_r - np.square(b_r))
    delta = np.square(b_mr) - 4*a_mr*c_mr

    #get x_mid_right and y_mid_right
    x_mr1 = (-b_mr - np.sqrt(delta)) / (2*a_mr+0.000001)
    x_mr2 = (-b_mr + np.sqrt(delta)) / (2*a_mr+0.000001)

    y_mr1 = a_r * x_mr1 + b_r
    y_mr2 = a_r * x_mr2 + b_r

    #return mid_line
    if x_mr1 > x_inter:
        return mid_line_equation(x_inter, y_inter, x_ml, y_ml, x_mr1, y_mr1)
    else:
        return mid_line_equation(x_inter, y_inter, x_ml, y_ml, x_mr2, y_mr2)

def lane_making(img):
    #import image
    #cover to hsv
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #blur
    line_mask = np.ones((len(img), len(img[0])))
    line_mask[:int(len(img)/3), :] = 0
    line_mask[-int(len(img)/20):, :] = 0

    line_img = line_mask*img_gray
    blur_img = cv2.GaussianBlur(line_img,(5, 5), 5)

    kernel1 = np.ones((5, 5), np.uint8)

    bw = cv2.threshold(blur_img,200,255,cv2.THRESH_BINARY_INV)[1]
    bw = cv2.erode(bw, kernel1, iterations=1)

    #find edges
    t_lower = 0  # Lower Threshold
    t_upper = 255 # Upper threshold

    # Applying the Canny Edge filter
    edges = cv2.Canny(np.uint8(bw), t_lower, t_upper)
    #cv2.imshow('Edge', bw)
    lines = cv2.HoughLines(edges, 1, np.pi / 50, 60, None, 0, 0)

    if lines is None:
        #cv2.putText(img, 'None', (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,0,255), 2)
        #cv2.imshow('Frame', img)
        return None, None, None, None
    else:
        len_line = len(lines)
        get_min = False
        get_max = False

        if len(lines) < 3:
            #cv2.putText(img, 'None', (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,0,255), 2)
            #cv2.imshow('Frame', img)
            return None, None, None, None
        else:
            #draw lines
            min_pt = [len(img[0]), 0, 0]
            max_pt = [0, 0, 0]

            #get 2 main lines
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

                #cv2.line(img, pt1, pt2, (100, 0, 255), 3, cv2.LINE_AA)

                if abs(pt1[1]-pt2[1]) >30:
                    if min_pt[0] > x0:
                        min_pt[0] = x0
                        min_pt[1] = pt1
                        min_pt[2] = pt2
                        get_min = True


                    if max_pt[0] < x0:
                        max_pt[0] = x0
                        max_pt[1] = pt1
                        max_pt[2] = pt2
                        get_max = True
                else:
                    len_line -= 1

            #get center line

            if len_line >= 2 and get_min and get_max:
                #get center

                center = detect_mid_line(min_pt[1], min_pt[2], max_pt[1], max_pt[2])
                x_inter = center[1]
                y_inter = center[0]
                a_mid = center[4]
                b_mid = center[5]

                if x_inter < 0.5 * len(img):
                    cv2.putText(img, ' y = {a}x + {b}'.format(a = round(a_mid,3), b = round(b_mid,3)), (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,0,255), 2)
                    cv2.line(img, min_pt[1], min_pt[2], (100,0,255), 3, cv2.LINE_AA)
                    cv2.line(img, max_pt[1], max_pt[2], (100,0,255), 3, cv2.LINE_AA)
                    cv2.line(img, (center[0], center[1]), (center[2], center[3]), (255,100,0), 3, cv2.LINE_AA)
                    #cv2.imshow('Frame', img)
                    return x_inter, y_inter, a_mid, b_mid
                else:
                    #cv2.putText(img, 'None', (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,0,255), 2)
                    #cv2.imshow('Frame', img)
                    return None, None, None, None
            else:
                #cv2.putText(img, 'None', (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,0,255), 2)
                #cv2.imshow('Frame', img)
                return None, None, None, None



def nextMove(imshape,leftSpeed, rightSpeed, leftfunction, rightfunction, intersection):
    goodinterval = [imshape[1] * 2/5, imshape[1] * 4/5]
    if leftfunction is None and rightfunction is None:
        rightSpeed = 0
        leftSpeed = 25
    elif leftfunction is None:
        rightSpeed = 25
        leftSpeed = 0
    elif  rightfunction is None:
        rightSpeed = 0
        leftSpeed = 35
    elif 240 < intersection[1]:
        rightSpeed = 0
        leftSpeed = 40
    elif intersection[0] < goodinterval[0]:
        rightSpeed = 35
        leftSpeed = 0
    elif intersection[0] > goodinterval[1]:
        rightSpeed = 0
        leftSpeed = 40
    elif goodinterval[0] < intersection[0] <goodinterval[1]:
        rightSpeed = 50
        leftSpeed = 50

    return leftSpeed, rightSpeed
