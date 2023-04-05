import Jetson.GPIO as GPIO
import timeit
import cv2
import numpy as np
from funct import *
import pygame
import math
import sys
in1 = 37 #left 
in2 = 35 #left //forward
in3 = 33 #right //forward
in4 = 31 #right 
leftSpeed = 120
rightSpeed = 120
GPIO.setmode(GPIO.BOARD)
GPIO.setup(in1,GPIO.OUT)
GPIO.setup(in2,GPIO.OUT)
GPIO.setup(in3,GPIO.OUT)
GPIO.setup(in4,GPIO.OUT)
GPIO.setwarnings(False)
capture = cv2.VideoCapture(0)
size = (int(capture.get(3)),int(capture.get(4)))
#result = cv2.VideoWriter('realrecord.mp4', 
#                         cv2.VideoWriter_fourcc(*'MP4V'),
#                         10, size)
now = timeit.default_timer()
fps,count = 0,0


pygame.init()
display = pygame.display.set_mode((300, 300))
sign_score = np.array([0, 0, 0])
_sign = ['Stop', 'Left', 'Right']
_max_red, _max_blue = 30000, 40000
ID = 0


while True:
    if ID == 1:
        ret,frame = capture.read()
        mid  = frame.shape[1] / 2
        sign = sign_detect(frame, min_red=8000, min_blue=8000)
        if len(_sign) == 3:
            
            sign_score += np.array(sign[0])

            if sign_score[0] == 8:
                _sign = ['Stop']
            if sign_score[1] == 8:
                _sign = ['Left']
            if sign_score[2] == 8:
                _sign = ['Right']
        else:
            if _sign[0] == 'Stop':
                if sign[1] > _max_red:
                    # Stop
                    GPIO.output(in1,0)
                    GPIO.output(in2,0)
                    GPIO.output(in3,0)
                    GPIO.output(in4,0)
                    time.sleep(3)
            else:
                if sign[1] > _max_blue:
                    if _sign[0] == 'Left':
                        GPIO.output(in1,0)
                        GPIO.output(in2,0)
                        GPIO.output(in3,0)
                        GPIO.output(in4,0)
                        time.sleep(2)
                        GPIO.output(in1,0)
                        GPIO.output(in2,20)
                        GPIO.output(in3)
                        GPIO.output(in4,0)
                        time.sleep(0.9)
                        GPIO.output(in1,0)
                        GPIO.output(in2,20)
                        GPIO.output(in3,20)
                        GPIO.output(in4,0)
                        time.sleep(0.9)
                    else:
                        #stop -> turn right
                        GPIO.output(in1,0)
                        GPIO.output(in2,0)
                        GPIO.output(in3,0)
                        GPIO.output(in4,0)
                        time.sleep(2)
                        GPIO.output(in1,0)
                        GPIO.output(in2,20)
                        GPIO.output(in3)
                        GPIO.output(in4,0)
                        time.sleep(0.9)
                        GPIO.output(in1,0)
                        GPIO.output(in2,20)
                        GPIO.output(in3,20)
                        GPIO.output(in4,0)
                        time.sleep(0.9)
                        print()

        print(_sign, sign[1])
        cv2.putText(frame, 'Sign: {st}'.format(st=str(sign)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 255), 2)

        leftfunction, rightfunction, intersection = bigway(frame,(210,210,210))

        # detect sign



        leftSpeed,rightSpeed  = nextMove(frame.shape,leftSpeed,rightSpeed, leftfunction,rightfunction,intersection)
        GPIO.output(in1,0)
        GPIO.output(in2,leftSpeed)
        GPIO.output(in3,rightSpeed)
        GPIO.output(in4,0)
        print("speed:",leftSpeed,"       ",rightSpeed)
        print("coordinate:",intersection)
        print("fps:",fps)
    else:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            # checking if keydown event happened or not
            if event.type == pygame.KEYDOWN:
               
                # checking if key "A" was pressed
                if event.key == pygame.K_w:
                    GPIO.output(in1,0)
                    GPIO.output(in2,255)
                    GPIO.output(in3,255)
                    GPIO.output(in4,0)
                # checking if key "J" was pressed
                if event.key == pygame.K_s:
                    GPIO.output(in1,255)
                    GPIO.output(in2,0)
                    GPIO.output(in3,0)
                    GPIO.output(in4,255)
                
                # checking if key "P" was pressed
                if event.key == pygame.K_a:
                    GPIO.output(in1,0)
                    GPIO.output(in2,0)
                    GPIO.output(in3,255)
                    GPIO.output(in4,0)
                
                # checking if key "M" was pressed
                if event.key == pygame.K_d:
                    GPIO.output(in1,0)
                    GPIO.output(in2,255)
                    GPIO.output(in3,0)
                    GPIO.output(in4,0)
                if event.key == pygame.K_LSHIFT:
                    GPIO.output(in1,0)
                    GPIO.output(in2,0)
                    GPIO.output(in3,0)
                    GPIO.output(in4,0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    count += 1
    if timeit.default_timer() - now >= 2:
        now = timeit.default_timer()
        fps = count /2
        count = 0
capture.release()
#result.release()
    
# Closes all the frames
cv2.destroyAllWindows()

